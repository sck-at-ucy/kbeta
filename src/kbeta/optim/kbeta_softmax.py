"""
Author: Stavros Kassinos
Date: April 2025
Version: b.0.0.1

FullySchedulableKourkoutas & FullySchedulableKourkoutasWithMomentum

Description
-----------
The Kourkoutas family of optimizers introduces a Bayesian-flavored update rule
where each parameter is associated with a 'posterior mean' and 'posterior variance.'
Inspired by a desert-lizard analogy, the optimizer 'scampers' through the parameter
space by combining new gradient information with stored memory—then throws in a bit
of random noise for exploration. Hyperparameters like 'sand_temperature,' 'desert_haze,'
and 'sunbathing' control how “hot and hazy” the learning environment is. Clipping terms
(rock_bottom, rock_ceiling, grad_clip) keep updates from ‘falling off the dunes,’ while
momentum (beta_m) can optionally give the lizard a push if you’re so inclined.

Kourkoutas can shine in problems with tricky nonconvex landscapes, especially PDEs or
small-data setups where a bit of exploratory stochasticity helps avoid poor local minima.
It may also do well in high-uncertainty tasks that benefit from “Bayesian-like” per-parameter
variance tracking. On the other hand, very large or well-conditioned tasks might find its
extra random noise unnecessary, making simpler mainstream optimizers (Adam, etc.) more direct.
Stiff PDEs that require ultra-stable updates or real-time data streams might also be tough
terrains for Kourkoutas unless carefully tuned.

Fun Tidbits
-----------
- “Kourkoutas” is a playful reference to a scaly desert explorer, evoking the
  sun-baked, unpredictable environment it thrives in—much like this optimizer’s
  noise-driven search.
- The 'FullySchedulable' variants reflect the ambition to allow runtime updating
  of key hyperparameters, akin to a desert climate that changes unpredictably.
- This code was spontaneously created but has already demonstrated comparable or
  slightly better performance than Adam in some PDE-based PINN tests.

Wrapping Up
-----------
- Likely to excel at:
  * PDE-based problems, small-data or nonconvex tasks, and problems needing
    robust exploration or “Bayesian-like” uncertainty handling.
- Likely to struggle at:
  * Very large, data-rich tasks that don’t need as much exploration.
  * Extremely stiff PDE constraints demanding very stable updates.

Refer to the classes below for usage details. They share a similar structure:
`init_single` sets up 'posterior_mean' and 'posterior_var,' then `apply_single`
performs the desert-themed parameter update. Momentum (beta_m) can be added in
the second class for smoothing out the randomness further.
"""

#from typing import Callable, List, Union
#import mlx.core as mx
#import mlx.optimizers as optim
#from mlx.optimizers import Adam
#from mlx.optimizers import Optimizer
#from mlx.utils import tree_flatten, tree_unflatten, tree_map



import mlx.core as mx
from mlx.optimizers import Optimizer                 # <- same base as before
from typing import Any, Callable, Dict, Optional


class KourkoutasSoftmaxFlex(Optimizer):
    """
    Adam variant with *layer‑wise* dynamic β₂ (“sun‑spike”) and optional
    soft‑max AMSGrad / trust‑region features.

    --------------------------------------------------------------------------
    QUICK REFERENCE
    --------------------------------------------------------------------------
    decay        – 0 < decay ≤ 1.  Soft leak of the `v_max` buffer
                   (classic AMSGrad ⇒ decay=None).
    max_ratio    – Trust‑region cap, applied as  |Δθ| ≤ lr · max_ratio.
    adaptive_tiny– Adds an *extra* tiny term that scales with ⟨|θ|⟩.
                   When False the classic Adam denominator √v + eps is used.

    All three knobs default to “off”, preserving vanilla (“violin”) behaviour.
    """

    # ─────────────────────────── initialisation ────────────────────────────
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2_max: float = 0.999,
        beta2_min: float = 0.9,
        eps: float = 1e-8,
        alpha: float = 0.9,
        *,
        # --- diagnostics toggle
        diagnostics: bool = False,
        # --- tiny constants -------------------------------------------------
        tiny_spike: float = 1e-8,      # only inside sun‑spike β₂ logic
        tiny_denom: float = 1e-8,      # only in the Adam denominator
        # --- optional features ---------------------------------------------
        decay: Optional[float] = None,     # soft‑max AMSGrad
        max_ratio: Optional[float] = None, # trust‑region clip
        adaptive_tiny: bool = False,       # scale tiny_denom with |θ|
        # --- bias‑correction & bookkeeping ---------------------------------
        bias_correction: str = "none",     # "none" | "beta2max" | "exact"
        warmup_steps: int = 0,
        layer_key_fn: Optional[Callable[[mx.array], Any]] = None,
        schedulers=None,
    ):
        super().__init__(schedulers=schedulers)
        self._diag = diagnostics

        # ----- (possibly) scheduled scalars --------------------------------
        self._maybe_schedule("learning_rate", learning_rate)
        self._maybe_schedule("alpha",          alpha)

        # ----- fixed hyper‑parameters -------------------------------------
        self.beta1     = beta1
        self.beta2_max = beta2_max
        self.beta2_min = beta2_min
        self.eps       = eps
        self.alpha     = alpha

        self.tiny_spike = tiny_spike
        self.tiny_denom = tiny_denom

        # ----- feature toggles --------------------------------------------
        self.decay         = decay
        self.max_ratio     = max_ratio
        self.adaptive_tiny = adaptive_tiny

        # ----- bias‑correction mode ---------------------------------------
        assert bias_correction in {"none", "beta2max", "exact"}
        self.bias_correction = bias_correction

        # ----- misc bookkeeping -------------------------------------------
        self.layer_key_fn   = layer_key_fn
        self.warmup_steps   = mx.array(int(warmup_steps), dtype=mx.int64)
        self.state["step"]  = mx.array(0, dtype=mx.int64)
        self.state["_layer_stats"] = {}

        # light‑weight diagnostics (float32 scalars)
        if self._diag:                             # allocate only if needed
            for name, init in [
                ("diag_max_ratio",    0.0),
                ("diag_denom_min",    1e9),
                ("diag_upd_norm_max", 0.0),
                ("diag_vhat_max",     0.0),
            ]:
                self.state[name] = mx.array(init, dtype=mx.float32)

    # ───────────────────── per‑tensor slot initialiser ──────────────────────
    def init_single(self, p: mx.array, st: dict):
        st["m"] = mx.zeros_like(p)
        st["v"] = mx.zeros_like(p)
        st["beta2_cumprod"] = mx.ones([], dtype=p.dtype)

        # allocate `v_max` only if the user enabled *either* knob
        if (self.decay is not None) or (self.max_ratio is not None):
            st["v_max"] = mx.zeros_like(p)

    # ─────────────────────────── main update ───────────────────────────────
    def apply_gradients(self, grads: dict, params: dict):
        if not self._initialized:
            self.init(grads)

        # update scheduled scalars (lr, alpha, …)
        for k, sched in self._schedulers.items():
            self.state[k] = sched(self.step)

        alpha = self.state["alpha"]
        self.state["step"] = self.step + 1

        # ---- gather per‑group statistics ----------------------------------
        from mlx.utils import tree_map
        buckets: Dict[Any, Dict[str, Any]] = {}

        def collect(g, p, st):
            if g is None:
                return p
            gid = self._layer_id(p)
            bkt = buckets.setdefault(
                gid, {"sum_sq": mx.zeros([], dtype=g.dtype), "items": []}
            )
            bkt["sum_sq"] = mx.stop_gradient(bkt["sum_sq"] + mx.sum(mx.square(g)))
            bkt["items"].append((g, p, st))
            return p

        tree_map(collect, grads, params, self.state)

        # ---- per‑bucket processing ----------------------------------------
        for gid, data in buckets.items():
            sum_sq, items = data["sum_sq"], data["items"]

            ls = self.state["_layer_stats"].setdefault(
                gid, {"grad_ema": mx.zeros([], dtype=sum_sq.dtype)}
            )

            g_norm = mx.sqrt(sum_sq)
            ls["grad_ema"] = alpha * ls["grad_ema"] + (1 - alpha) * g_norm
            g_ema = ls["grad_ema"]

            # dynamic β₂ (“sun‑spike”)
            cond = (self.step <= self.warmup_steps).astype(g_norm.dtype)
            raw  = g_norm / (g_ema + self.tiny_spike)
            sun  = (1 - cond) * (raw / (1 + raw))

            beta2 = (
                cond * 0.5 * (self.beta2_min + self.beta2_max)
                + (1 - cond) * (self.beta2_max -
                                (self.beta2_max - self.beta2_min) * sun)
            )

            lr = self.learning_rate.astype(sum_sq.dtype)
            b1 = self.beta1
            
            if self._diag:
                ls["last_spike"] = sun
                ls["last_beta2"] = beta2

            # ---- per‑tensor inner loop ------------------------------------
            for g, p, st in items:
                m = st["m"] = b1 * st["m"] + (1 - b1) * g
                v = st["v"] = beta2 * st["v"] + (1 - beta2) * mx.square(g)

                # choose v̂ (plain, AMSGrad, or soft‑max)
                v_hat = v
                if "v_max" in st:
                    if self.decay is not None:             # soft‑leak
                        st["v_max"] = mx.maximum(mx.array(self.decay, v.dtype) * st["v_max"], v)
                    else:                                  # hard AMSGrad
                        st["v_max"] = mx.maximum(st["v_max"], v)
                    v_hat = st["v_max"]

                # tiny term for the denominator
                if self.adaptive_tiny:
                    tiny_local = self.tiny_denom * mx.maximum(mx.mean(mx.abs(p)), 1.0)
                else:
                    tiny_local = mx.array(0.0, dtype=v.dtype)

                # bias correction flavour
                if self.bias_correction == "none":
                    denom = mx.sqrt(v_hat) + tiny_local + self.eps
                    upd   = lr * m / denom
                else:
                    st["beta2_cumprod"] *= beta2
                    bc1 = 1.0 - b1 ** self.step
                    bc2 = (
                        1.0 - st["beta2_cumprod"]
                        if self.bias_correction == "exact"
                        else 1.0 - self.beta2_max ** self.step
                    )
                    denom = mx.sqrt(v_hat / bc2) + tiny_local + self.eps
                    upd   = (lr * (m / bc1)) / denom

                # optional trust‑region clip
                if self.max_ratio is not None:
                    lim = lr * self.max_ratio
                    upd = mx.clip(upd, -lim, lim)

                # diagnostics (scalar fast‑paths)
                if self._diag:
                
                    self.state["diag_max_ratio"]    = mx.maximum(
                        self.state["diag_max_ratio"], mx.max(mx.abs(upd) / lr)
                    )
                    self.state["diag_denom_min"]    = mx.minimum(
                        self.state["diag_denom_min"], mx.min(denom)
                    )
                    self.state["diag_upd_norm_max"] = mx.maximum(
                        self.state["diag_upd_norm_max"], mx.max(mx.abs(upd))
                    )
                    self.state["diag_vhat_max"]     = mx.maximum(
                        self.state["diag_vhat_max"],  mx.max(v_hat)
                    )

                # parameter update
                p -= upd

        return params

    # ───────────────────────── helpers / utilities ─────────────────────────
    def _layer_id(self, p: mx.array):
        if self.layer_key_fn is not None:
            return self.layer_key_fn(p)
        if getattr(p, "name", None):
            return p.name.split(":")[0]
        return id(p)

    @property
    def step(self):
        return self.state["step"]

    # light snapshot for training‑loop printouts
    def snapshot_diagnostics(self):
        keys = [
            "diag_denom_min",
            "diag_max_ratio",
            "diag_upd_norm_max",
            "diag_vhat_max",
        ]
        return {k: float(self.state[k].item()) for k in keys}

    def snapshot_sunspike_history(self):
        """
        Collect the most‑recent sun‑spike and β₂ value per layer *as Python floats*.
    
        Returns
        -------
        (spikes, betas) : tuple[list[float], list[float]]
            spikes[i] and betas[i] correspond to the same layer.
            Returns ([], []) if diagnostics are disabled.
        """
        if not self._diag:                      # flag set in __init__()
            return [], []
    
        spikes, betas = [], []
        for st in self.state["_layer_stats"].values():
            if "last_spike" in st:              # present only when diagnostics on
                spikes.append(float(st["last_spike"].item()))
                betas.append(float(st["last_beta2"].item()))
    
        return spikes, betas
