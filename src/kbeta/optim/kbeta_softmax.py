"""
Kourkoutas‑β — An Adam‑style optimiser with dynamic β₂ (“sun‑spike”) logic
=========================================================================

Author   : Stavros Kassinos
First rev: Apr 2025  –  This rev: Aug 2025 (“softmax‑flex” release)

-------------------------------------------------------------------------------
Desert‑lizard intuition ☀️🦎
-------------------------------------------------------------------------------
Picture a **Kourkoutas**, the quick silver-gold lizard endemic to Cyprus.

* **Blazing noon – Sun spikes**
  The ground is scorching, the lizard darts erratically to keep its feet cool.
  ⇒ Gradient variance is **high** → β₂ is *lowered* so the optimiser reacts
  faster (less momentum smoothing, more exploration).

* **Mild morning / dusk**
  The sand is cool, the lizard moves in long, measured strides.
  ⇒ Gradient variance is **low** → β₂ gravitates toward **β₂₍max₎**
  (behaviour converges to vanilla Adam for steady refinement).

Implementation
~~~~~~~~~~~~~~
For each layer we keep an EWMA of the gradient norm, `grad_ema`.

A “sun‑spike” scalar, using a bounded squash:

    raw = ‖g‖ / (grad_ema + tiny_spike)      ∈ (0, ∞)
    sun = raw / (1 + raw)                    ∈ [0, 1)

During warm‑up (step ≤ warmup_steps), we hold sun = 0 and set
β₂ = ½(β₂_min + β₂_max). After warm‑up:

    β₂ = β₂_max − (β₂_max − β₂_min) · sun

Low *sun* ⇒ β₂ ≈ β₂_max (conservative)
High *sun* ⇒ β₂ ≈ β₂_min (agile)

-------------------------------------------------------------------------------
Key additions over Adam / AMSGrad
-------------------------------------------------------------------------------
• Layer‑wise sun‑spike β₂ (see above).
• **Soft‑max AMSGrad** (`decay∈(0,1]`) – gently leaks the running v‑max buffer.
• **Trust‑region clip** (`max_ratio`) – caps |Δθ| ≤ lr·max_ratio.
• **Adaptive tiny term** (`adaptive_tiny`) – scales *eps* with ⟨|θ|⟩.
• **Diagnostics toggle** (`diagnostics`) – ultra‑cheap per‑epoch stats
  for plotting (sun‑spike, β₂, denom min, etc.).

All toggles default to **off**, so *Kourkoutas‑β collapses to Adam* when
`beta2_min == beta2_max == 0.999` and extras are disabled.

-------------------------------------------------------------------------------
When to try it
-------------------------------------------------------------------------------
✅ PDE & physics‑informed nets  ✅ small / noisy data  ✅ spiky gradients
❌ Huge, well‑conditioned vision/LN tasks where plain Adam already excels

-------------------------------------------------------------------------------
Quick‑start snippets
-------------------------------------------------------------------------------
**PINN setting**

```python
from kbeta.optim import KourkoutasSoftmaxFlex as Kβ

opt = Kβ(
    learning_rate = lr_schedule,
    beta1         = 0.90,
    beta2_max     = 0.999,                 # calm coasting
    beta2_min     = 0.88,                  # agile under spikes
    eps           = 1e-8,
    alpha         = 0.93,                  # EWMA for grad_ema
    tiny_spike    = 1e-9,
    tiny_denom    = 1e-8,
    decay         = 0.98,                  # soft‑max AMSGrad
    adaptive_tiny = True,
    max_ratio     = 3,
    bias_correction = "beta2max",
    layer_key_fn  = lambda p: p.shape,
    diagnostics   = True,                  # enables snapshot helpers
)
```

**Transformer setting**

```python
from kbeta.optim import KourkoutasSoftmaxFlex as Kβ

opt = Kβ(
    learning_rate = 1e-3,
    beta1         = 0.90,
    beta2_max     = 0.999,                  # calm coasting
    beta2_min     = 0.88,                   # agile under spikes
    eps           = 1e-8,
    alpha         = 0.93,
    adaptive_tiny = False,                  # often off for Transformer stacks
    layer_key_fn  = lambda p: p.shape,
    warmup_steps  = 350,
    diagnostics   = ARGS.kour_diagnostics,  # enables snapshot helpers
)
```

Inside your training loop you can call
spikes, betas = opt.snapshot_sunspike_history()
to feed violin/heat‑map plots.

Happy scurrying!  – Stavros
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx.optimizers import Optimizer  # <- same base as before


class KourkoutasSoftmaxFlex(Optimizer):
    """
    Adam variant with *layer‑wise* dynamic β₂ (“sun‑spike”) and optional
    soft‑max AMSGrad / trust‑region features.

    --------------------------------------------------------------------------
    QUICK REFERENCE
    --------------------------------------------------------------------------
    decay        – None: disable AMSGrad (unless max_ratio implies v_max);
                   1.0 : hard AMSGrad (non‑decreasing v_max);
                   (0,1): leaky‑AMSGrad (soft‑max bound);
                   0.0 : degenerate (v̂_t = v_t).
    max_ratio    – Trust‑region cap, applied as |Δθ| ≤ lr·max_ratio.
    adaptive_tiny– Adds an extra tiny term that scales with ⟨|θ⟩.
                   When False the classic Adam denominator √v + eps is used.

    All three knobs default to “off”, preserving vanilla Adam when
    decay=None, max_ratio=None, adaptive_tiny=False and β₂ is fixed.
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
        tiny_spike: float = 1e-8,  # only inside sun‑spike β₂ logic
        tiny_denom: float = 1e-8,  # only in the Adam denominator
        # --- optional features ---------------------------------------------
        decay: float | None = None,  # soft‑max AMSGrad
        max_ratio: float | None = None,  # trust‑region clip
        adaptive_tiny: bool = False,  # scale tiny_denom with |θ|
        # --- bias‑correction & bookkeeping ---------------------------------
        bias_correction: str = "none",  # "none" | "beta2max" | "exact"
        warmup_steps: int = 0,
        layer_key_fn: Callable[[mx.array], Any] | None = None,
        schedulers=None,
    ):
        super().__init__(schedulers=schedulers)
        self._diag = diagnostics

        # ----- (possibly) scheduled scalars --------------------------------
        self._maybe_schedule("learning_rate", learning_rate)
        self._maybe_schedule("alpha", alpha)

        # ----- fixed hyper‑parameters -------------------------------------
        self.beta1 = beta1
        self.beta2_max = beta2_max
        self.beta2_min = beta2_min
        self.eps = eps
        self.alpha = alpha

        self.tiny_spike = tiny_spike
        self.tiny_denom = tiny_denom

        # ----- feature toggles --------------------------------------------
        self.decay = decay
        self.max_ratio = max_ratio
        self.adaptive_tiny = adaptive_tiny

        # ----- bias‑correction mode ---------------------------------------
        assert bias_correction in {"none", "beta2max", "exact"}
        self.bias_correction = bias_correction

        # ----- misc bookkeeping -------------------------------------------
        self.layer_key_fn = layer_key_fn
        self.warmup_steps = mx.array(int(warmup_steps), dtype=mx.int64)
        self.state["step"] = mx.array(0, dtype=mx.int64)
        self.state["_layer_stats"] = {}

        # light‑weight diagnostics (float32 scalars)
        if self._diag:  # allocate only if needed
            for name, init in [
                ("diag_max_ratio", 0.0),
                ("diag_denom_min", 1e9),
                ("diag_upd_norm_max", 0.0),
                ("diag_vhat_max", 0.0),
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

        buckets: dict[Any, dict[str, Any]] = {}

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
            raw = g_norm / (g_ema + self.tiny_spike)
            sun = (1 - cond) * (raw / (1 + raw))

            beta2 = cond * 0.5 * (self.beta2_min + self.beta2_max) + (1 - cond) * (
                self.beta2_max - (self.beta2_max - self.beta2_min) * sun
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
                    if self.decay is not None:  # soft‑leak
                        st["v_max"] = mx.maximum(
                            mx.array(self.decay, v.dtype) * st["v_max"], v
                        )
                    else:  # hard AMSGrad
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
                    upd = lr * m / denom
                else:
                    st["beta2_cumprod"] *= beta2
                    bc1 = 1.0 - b1**self.step
                    bc2 = (
                        1.0 - st["beta2_cumprod"]
                        if self.bias_correction == "exact"
                        else 1.0 - self.beta2_max**self.step
                    )
                    denom = mx.sqrt(v_hat / bc2) + tiny_local + self.eps
                    upd = (lr * (m / bc1)) / denom

                # optional trust‑region clip
                if self.max_ratio is not None:
                    lim = lr * self.max_ratio
                    upd = mx.clip(upd, -lim, lim)

                # diagnostics (scalar fast‑paths)
                if self._diag:
                    self.state["diag_max_ratio"] = mx.maximum(
                        self.state["diag_max_ratio"], mx.max(mx.abs(upd) / lr)
                    )
                    self.state["diag_denom_min"] = mx.minimum(
                        self.state["diag_denom_min"], mx.min(denom)
                    )
                    self.state["diag_upd_norm_max"] = mx.maximum(
                        self.state["diag_upd_norm_max"], mx.max(mx.abs(upd))
                    )
                    self.state["diag_vhat_max"] = mx.maximum(
                        self.state["diag_vhat_max"], mx.max(v_hat)
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
        if not self._diag:  # flag set in __init__()
            return [], []

        spikes, betas = [], []
        for st in self.state["_layer_stats"].values():
            if "last_spike" in st:  # present only when diagnostics on
                spikes.append(float(st["last_spike"].item()))
                betas.append(float(st["last_beta2"].item()))

        return spikes, betas
