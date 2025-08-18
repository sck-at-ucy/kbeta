#!/usr/bin/env python3

"""
Testbed D — Minimal MLX character-level Transformer training script.

Purpose
-------
Small, dependency-light example used in the paper to compare **K‑β** against
Adam on a next‑token character task. It keeps the model tiny and the data
pipeline deterministic so optimizer differences are easy to see.

Highlights
----------
- Deterministic data pipeline (NumPy RNG) and fixed validation batch.
- Optional JIT compile of loss+grad for each (B, L) shape.
- "Layer buckets" that map parameters to optimizer groups for K‑β:
  * global (all params together)
  * shape (group by array shape)
  * per-array (stable per-parameter IDs across steps)
- Decoupled weight decay (AdamW‑style) applied **outside** the optimizer.
- LR schedule via step:value pairs (e.g. "0:3e-4,20000:1e-4").

Quick reference
---------------
Training with K‑β (per‑array buckets, compiled, 50k steps):

    python examples/testbed_d.py \\
        --text data/tinyshakespeare.txt \\
        --steps 50000 --batch 64 --ctx 256 \\
        --opt kbeta --lr 3e-4 --warmup 200 \\
        --layer_bucket per-array --compile --len_bucket 32 \\
        --eval_every 2000

Training with Adam (β₂=0.95):

    python examples/testbed_d.py --text data/tinyshakespeare.txt \\
        --opt adam --adam_beta2 0.95

Reproducibility
---------------
- Model init and MLX internals use `--seed`.
- Batching and eval batches use **NumPy** RNGs with fixed seeds to keep
  draws stable across runs.
- The `--fixed_eval_seed` controls the held‑out validation slice.

Notes
-----
- The script parses `--early_stop_*` flags for parity with other benches,
  but **does not implement early stopping** in this minimal example.
"""

import argparse
import math
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

try:
    from kbeta.optim import KourkoutasSoftmaxFlex as Kbeta

    HAVE_KBETA = True
except Exception:
    HAVE_KBETA = False
from mlx.optimizers import Adam as MLXAdam

# ------------------------------- helpers -------------------------------


def gelu(x):
    """Tanh-based GELU approximation used in GPT-style MLPs.

    Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    return (
        0.5
        * x
        * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
    )


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dimension, using explicit mean/var.

    Args:
        x: (..., C) tensor.
        weight, bias: affine parameters of shape (C,).
        eps: numerical stability epsilon.
    """
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    xhat = (x - mean) / mx.sqrt(var + eps)
    return xhat * weight + bias


def causal_mask(T, dtype):
    """Return a (T, T) lower-triangular causal mask of the given dtype."""
    i = mx.arange(T)
    m = i[None, :] <= i[:, None]
    return m.astype(dtype)


def cross_entropy_logits(logits, targets):
    """Mean token-level NLL from unnormalized logits.

    Args:
        logits: (B, T, V)
        targets: (B, T) int32 with token indices in [0, V).

    Returns:
        Scalar mean loss over all positions/tokens.
    """
    lse = mx.logsumexp(logits, axis=-1)  # (B, T)
    tgt = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return mx.mean(lse - tgt)


# ------------------------------- model -------------------------------


@dataclass
class Config:
    """Transformer configuration (kept tiny on purpose)."""

    vocab: int
    ctx: int
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 6
    lmin: int = 64
    lmax: int = 256


def init_params(cfg: Config, rng_seed: int = 0):
    """Initialize Transformer weights with small Gaussian noise.

    Returns:
        Dict tree of MX arrays; keys follow a simple, flat naming scheme.
    """
    mx.random.seed(int(rng_seed))
    p = {}
    p["tok_emb"] = 0.02 * mx.random.normal((cfg.vocab, cfg.d_model))
    p["pos_emb"] = 0.02 * mx.random.normal((cfg.ctx, cfg.d_model))
    for i in range(cfg.n_layer):
        b = {}
        b["ln1_w"] = mx.ones((cfg.d_model,))
        b["ln1_b"] = mx.zeros((cfg.d_model,))
        b["w_qkv"] = 0.02 * mx.random.normal((cfg.d_model, 3 * cfg.d_model))
        b["b_qkv"] = mx.zeros((3 * cfg.d_model,))
        b["w_o"] = 0.02 * mx.random.normal((cfg.d_model, cfg.d_model))
        b["b_o"] = mx.zeros((cfg.d_model,))
        b["ln2_w"] = mx.ones((cfg.d_model,))
        b["ln2_b"] = mx.zeros((cfg.d_model,))
        hidden = 4 * cfg.d_model
        b["w_fc1"] = 0.02 * mx.random.normal((cfg.d_model, hidden))
        b["b_fc1"] = mx.zeros((hidden,))
        b["w_fc2"] = 0.02 * mx.random.normal((hidden, cfg.d_model))
        b["b_fc2"] = mx.zeros((cfg.d_model,))
        p[f"block_{i}"] = b
    p["lnf_w"] = mx.ones((cfg.d_model,))
    p["lnf_b"] = mx.zeros((cfg.d_model,))
    p["head_w"] = 0.02 * mx.random.normal((cfg.d_model, cfg.vocab))
    p["head_b"] = mx.zeros((cfg.vocab,))
    return p


def self_attention(x, w_qkv, b_qkv, w_o, b_o, n_head, mask_bool):
    """Single masked self-attention layer (no dropout/flash tricks).

    Args:
        x: (B, T, C)
        w_qkv, b_qkv: project to (Q, K, V).
        w_o, b_o: output projection back to C.
        n_head: number of attention heads.
        mask_bool: (T, T) boolean causal mask (True=keep).

    Returns:
        Updated hidden states of shape (B, T, C).
    """
    B, T, C = x.shape
    H = n_head
    D = C // H
    qkv = x @ w_qkv + b_qkv
    q, k, v = mx.split(qkv, 3, axis=-1)

    def split_heads(t):
        t = t.reshape((B, T, H, D))
        return mx.transpose(t, (0, 2, 1, 3))

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    att = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(D)
    neg_inf = mx.array(-1e9, dtype=att.dtype)
    att = mx.where(mask_bool[None, None, :, :], att, neg_inf)
    att = mx.softmax(att, axis=-1)
    y = att @ v
    y = mx.transpose(y, (0, 2, 1, 3)).reshape((B, T, C))
    y = y @ w_o + b_o
    return y


def transformer_forward(params, cfg: Config, idx):
    """Forward pass for the tiny Transformer.

    Args:
        params: dict tree from `init_params`.
        cfg: Config with model hyperparameters.
        idx: (B, T) int32 token indices; T <= cfg.ctx.

    Returns:
        Logits of shape (B, T, V).
    """
    B, T = idx.shape
    assert T <= cfg.ctx, "sequence length exceeds context window"
    tok = mx.take(params["tok_emb"], idx, axis=0)  # (B,T,C)
    pos = mx.take(params["pos_emb"], mx.arange(T), axis=0)  # (T,C)
    x = tok + pos[None, :, :]
    cmask = causal_mask(T, dtype=mx.bool_)
    for i in range(cfg.n_layer):
        b = params[f"block_{i}"]
        xn = layer_norm(x, b["ln1_w"], b["ln1_b"])
        xa = self_attention(
            xn, b["w_qkv"], b["b_qkv"], b["w_o"], b["b_o"], cfg.n_head, cmask
        )
        x = x + xa
        xn = layer_norm(x, b["ln2_w"], b["ln2_b"])
        xm = gelu(xn @ b["w_fc1"] + b["b_fc1"])
        xm = xm @ b["w_fc2"] + b["b_fc2"]
        x = x + xm
    x = layer_norm(x, params["lnf_w"], params["lnf_b"])
    logits = x @ params["head_w"] + params["head_b"]
    return logits


# ------------------------------- data -------------------------------


def load_text(path):
    """Read a UTF‑8 text file into a single Python string."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    """Build a simple character vocabulary and encode the text.

    Returns:
        stoi: dict char -> id
        itos: dict id -> char
        ids:  mx.int32 array of token ids for `text`
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = mx.array([stoi[c] for c in text], dtype=mx.int32)
    return stoi, itos, ids


# deterministic indexer (no RNG here)
def index_batch(ids, starts, L):
    """Slice `ids` at deterministic start positions into (x, y) pairs.

    Args:
        ids:   (N,) token id vector.
        starts:(B,) starting offsets.
        L:     int sequence length.

    Returns:
        x, y:  (B, L) where y is x shifted by +1.
    """
    L = int(L)
    offs = mx.arange(L)[None, :]
    idx = starts[:, None] + offs
    x = mx.take(ids, idx)
    y = mx.take(ids, idx + 1)
    return x, y


# ----------------------------- optimizers -----------------------------


def make_kbeta(lr, warmup_steps, layer_bucket="global", layer_key_fn=None):
    """Construct a K‑β optimizer with the paper's default knobs.

    Notes:
        - `layer_key_fn` maps a parameter array -> bucket key.
        - When `layer_bucket='per-array'`, we pass a stable per-parameter key.
    """
    if not HAVE_KBETA:
        raise RuntimeError("KourkoutasSoftmaxFlex not available")
    return Kbeta(
        learning_rate=lr,
        beta1=0.9,
        beta2_max=0.999,
        beta2_min=0.88,
        eps=1e-8,
        alpha=0.93,
        tiny_spike=1e-9,
        tiny_denom=1e-8,
        decay=None,
        max_ratio=None,
        adaptive_tiny=False,
        # decay=0.98, max_ratio = 3,
        bias_correction="beta2max",
        warmup_steps=warmup_steps,
        layer_key_fn=layer_key_fn,
        diagnostics=False,
    )


def make_adam(lr, beta2=0.999):
    """MLX Adam wrapper (bias correction on by default)."""
    return MLXAdam(learning_rate=lr, betas=[0.9, beta2], eps=1e-8, bias_correction=True)


def set_lr(opt, lr):
    """Set the learning rate on either MLX or K‑β optimizers (if present)."""
    if hasattr(opt, "learning_rate"):
        opt.learning_rate = float(lr)


# --------------------------- LR schedule utils ------------------------


def parse_lr_schedule(spec: str):
    """Parse 'step:value' comma lists into sorted [(step, value), ...]."""
    pairs = []
    for part in spec.split(","):
        s, v = part.strip().split(":")
        pairs.append((int(s), float(v)))
    return sorted(pairs, key=lambda t: t[0])


def lr_at(step, base_lr, schedule):
    """Return the LR active at a given step under a piecewise-constant schedule."""
    lr = base_lr
    for s, v in schedule:
        if step >= s:
            lr = v
        else:
            break
    return lr


# --------------------------- decoupled weight decay -------------------


def _walk_params_inplace(d, fn, prefix=""):
    """Recursively apply `fn(path, leaf)` to a nested dict in place."""
    # Recursively apply fn(path, leaf) -> new_leaf; mutate in place
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _walk_params_inplace(v, fn, path)
        else:
            d[k] = fn(path, v)


def _is_bias(path):
    """Heuristic: treat names ending with '_b' or starting with 'b_' as biases."""
    name = path.split(".")[-1]
    return name.endswith("_b") or name.startswith("b_")


def _is_layernorm(path):
    """True for LayerNorm parameter names: 'ln1_*', 'ln2_*', 'lnf_*'."""
    # catches 'ln1_w', 'ln1_b', 'ln2_*', 'lnf_*'
    name = path.split(".")[-1]
    return "ln" in name


def _is_embedding(path):
    """True for top-level token/position embeddings."""
    # top-level tok_emb / pos_emb
    base = path.split(".")[0]
    return base in ("tok_emb", "pos_emb")


def apply_decoupled_weight_decay(
    params, lr, wd, decay_bias=False, decay_norm=False, decay_embed=False
):
    """Apply AdamW-style decay *outside* the optimizer by scaling selected tensors.

    Scales eligible leaves in `params` by (1 - lr*wd) in place.

    Args:
        params: parameter dict tree (mutated in place).
        lr: current learning rate (float).
        wd: weight decay coefficient.
        decay_bias / decay_norm / decay_embed: include those groups if True.

    Returns:
        Number of tensors actually decayed (int). Returns 0 if wd<=0 or lr<=0.
    """
    """Scale selected parameters by (1 - lr*wd) in-place (AdamW-style, outside the optimizer)."""
    if wd <= 0 or lr <= 0:
        return 0  # nothing done
    scale = 1.0 - float(lr) * float(wd)

    def maybe_decay(path, arr):
        if (not decay_bias) and _is_bias(path):
            return arr
        if (not decay_norm) and _is_layernorm(path):
            return arr
        if (not decay_embed) and _is_embedding(path):
            return arr
        return arr * scale

    # Count how many arrays actually decayed
    count = 0

    def count_and_decay(path, arr):
        nonlocal count
        decays = True
        if (not decay_bias) and _is_bias(path):
            decays = False
        if (not decay_norm) and _is_layernorm(path):
            decays = False
        if (not decay_embed) and _is_embedding(path):
            decays = False
        if decays:
            count += 1
        return maybe_decay(path, arr)

    _walk_params_inplace(params, count_and_decay)
    return count


# ------------------------------- train -------------------------------


def main():
    """CLI entry point.

    The default hyperparameters match the tiny Transformer used in the paper.
    This script is designed for clarity rather than throughput; see the
    comments near `--compile`, `--len_bucket`, and `--barrier_every` for
    knobs that affect compile behavior and execution barriers.

    Tip: when benchmarking optimizers, keep `--text`, `--seed`,
    and the LR schedule identical across runs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--lmin", type=int, default=64)
    ap.add_argument("--lmax", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt", type=str, default="kbeta", choices=["kbeta", "adam"])
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument(
        "--layer_bucket",
        type=str,
        default="global",
        choices=["global", "shape", "per-array"],
    )
    # compile / performance
    ap.add_argument(
        "--compile",
        action="store_true",
        help="JIT-compile loss+grad for each (B,L) shape",
    )
    ap.add_argument(
        "--len_bucket",
        type=int,
        default=32,
        help="round L up to a multiple to reduce compile churn; 1 disables",
    )
    ap.add_argument(
        "--lr_schedule",
        type=str,
        default="",
        help='e.g. "0:3e-4,20000:1.5e-4,40000:1e-4,50000:1e-5"',
    )
    ap.add_argument(
        "--barrier_every",
        type=int,
        default=1,
        help="call mx.eval on params (and optimizer state if exposed) every N steps (default: 1)",
    )
    # eval / reproducibility
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="fraction of the text held out for validation",
    )
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument(
        "--fixed_eval_seed", type=int, default=1234, help="freeze the eval batch"
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="use a stable per-array bucket and deterministic length draw",
    )
    # NEW: decoupled weight decay (AdamW-style), outside the optimizer
    ap.add_argument(
        "--wd", type=float, default=0.0, help="decoupled weight decay; 0 disables"
    )
    ap.add_argument(
        "--wd_bias", action="store_true", help="also decay biases (default: off)"
    )
    ap.add_argument(
        "--wd_norm",
        action="store_true",
        help="also decay LayerNorm params (default: off)",
    )
    ap.add_argument(
        "--wd_embed",
        action="store_true",
        help="also decay tok/pos embeddings (default: off)",
    )
    # Parsed for parity / future use; not implemented in this minimal example.
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="patience in number of eval checks with no improvement; 0 disables",
    )
    ap.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="min absolute improvement in eval loss to reset patience",
    )
    ap.add_argument(
        "--early_stop_warmup",
        type=int,
        default=0,
        help="number of initial eval checks to ignore before early-stop logic",
    )
    args = ap.parse_args()

    # keep MLX RNG seeded for param init etc., but NOT for batching
    mx.random.seed(int(args.seed))

    text = load_text(args.text)
    stoi, itos, ids = build_vocab(text)
    V = len(stoi)
    print(f"[data] {len(text):,} chars | vocab={V}")

    cfg = Config(
        vocab=V,
        ctx=args.ctx,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        lmin=args.lmin,
        lmax=args.lmax,
    )

    params = init_params(cfg, rng_seed=args.seed)

    # --- stable per-array bucketing: maintain id->stable-index map and refresh each step
    leaf_map = {}

    def rebuild_leaf_map():
        """Recompute id->index for current param arrays (ids can change after updates)."""
        # Recompute id->index for *current* param arrays (ids change after each optimizer step)
        leaf_map.clear()
        for i, (_, v) in enumerate(tree_flatten(params)):
            leaf_map[id(v)] = i

    def per_array_stable(p):
        """Stable integer key per parameter leaf (used when layer_bucket='per-array')."""
        # Stable integer per parameter path (refreshed via rebuild_leaf_map())
        return leaf_map.get(id(p), -1)

    # Choose layer_key_fn for K-β based on layer_bucket
    layer_key_fn = None
    if args.layer_bucket == "global":
        layer_key_fn = lambda _: 0  # noqa: E731
    elif args.layer_bucket == "shape":
        layer_key_fn = lambda p: p.shape  # noqa: E731
    elif args.layer_bucket == "per-array":
        layer_key_fn = per_array_stable
        rebuild_leaf_map()  # ensure mapping is correct before first optimizer call

    if args.opt == "kbeta":
        if not HAVE_KBETA:
            print("[WARN] kbeta not available; falling back to Adam")
            opt = make_adam(args.lr, beta2=args.adam_beta2)
        else:
            opt = make_kbeta(
                args.lr,
                args.warmup,
                layer_bucket=args.layer_bucket,
                layer_key_fn=layer_key_fn,
            )
            print(
                "[opt] Using K-β: beta2 ∈ [0.88, 0.999], alpha=0.93, warmup, bias_correction='beta2max'"
            )
            print(
                f"[opt] layer_bucket={args.layer_bucket} ({'stable per-parameter buckets' if args.layer_bucket=='per-array' else 'pooled'})"
            )
    else:
        opt = make_adam(args.lr, beta2=args.adam_beta2)
        print(f"[opt] Using MLX Adam (β2={args.adam_beta2}, bias-correction on)")

    if args.wd > 0:
        excl = []
        if not args.wd_bias:
            excl.append("bias")
        if not args.wd_norm:
            excl.append("LayerNorm")
        if not args.wd_embed:
            excl.append("embeddings")
        print(
            f"[wd] Decoupled weight decay enabled: wd={args.wd:g}; excluding {', '.join(excl) if excl else 'nothing'}"
        )

    schedule = parse_lr_schedule(args.lr_schedule) if args.lr_schedule else []

    # ---------------- train/val split + fixed eval batch ----------------
    N = ids.shape[0]
    split = int(N * (1.0 - args.val_frac))
    train_ids = ids[:split]
    val_ids = ids[split:]
    L_eval = min(256, args.ctx)

    # Deterministic eval batch via NumPy RNG (independent of MLX RNG)
    rng_eval = np.random.default_rng(args.fixed_eval_seed)
    high_eval = int(val_ids.shape[0] - (L_eval + 1))
    if high_eval <= 0:
        raise ValueError(
            f"L_eval={L_eval} too large for validation set of size {int(val_ids.shape[0])}"
        )
    starts_eval = rng_eval.integers(
        0, high_eval + 1, size=(args.eval_bs,), dtype=np.int32
    )
    xe_fixed, ye_fixed = index_batch(
        val_ids, mx.array(starts_eval, dtype=mx.int32), L_eval
    )

    # --------- compiled loss+grad (optional) ----------
    def _loss_for_grad(p, x, y):
        """Helper so we can take gradients w.r.t. params and reuse the loss."""
        logits = transformer_forward(p, cfg, x)
        return cross_entropy_logits(logits, y)

    if args.compile:

        @mx.compile
        def loss_and_grad(p, x, y):
            """Return (grad_tree, loss_scalar) with optional JIT."""
            g = mx.grad(_loss_for_grad)(p, x, y)
            return g, _loss_for_grad(p, x, y)

    else:

        def loss_and_grad(p, x, y):
            """Return (grad_tree, loss_scalar) without JIT."""
            g = mx.grad(_loss_for_grad)(p, x, y)
            return g, _loss_for_grad(p, x, y)

    # --------- deterministic length + start schedules for training ---------
    # Lengths:
    if args.deterministic:
        # fixed L per step (bucketed if requested)
        if args.len_bucket > 1:
            L_fix = (
                (args.lmax + args.len_bucket - 1) // args.len_bucket
            ) * args.len_bucket
        else:
            L_fix = args.lmax
        L_seq = np.full((args.steps,), min(args.ctx, L_fix), dtype=np.int32)
    else:
        rng_len = np.random.default_rng(args.seed + 17)
        rawL = rng_len.integers(
            low=args.lmin, high=args.lmax + 1, size=args.steps, dtype=np.int32
        )
        if args.len_bucket > 1:
            rawL = ((rawL + args.len_bucket - 1) // args.len_bucket) * args.len_bucket
        L_seq = np.minimum(rawL, args.ctx).astype(np.int32)

    # Starts per step:
    rng_b = np.random.default_rng(args.seed + 23)
    Ntr = int(train_ids.shape[0])
    starts_mat = np.empty((args.steps, args.batch), dtype=np.int32)
    for s in range(args.steps):
        Ls = int(L_seq[s])
        high = Ntr - (Ls + 1)
        if high <= 0:
            raise ValueError(f"L={Ls} too large for training set of size {Ntr}")
        starts_mat[s] = rng_b.integers(0, high + 1, size=(args.batch,), dtype=np.int32)

    # Warm-up a couple of shapes to populate caches (deterministic too)
    rng_warm = np.random.default_rng(args.seed + 101)
    for Lw in (args.lmin, min(args.lmax, args.ctx)):
        high_w = int(ids.shape[0] - (Lw + 1))
        if high_w <= 0:
            continue
        starts_w = rng_warm.integers(0, high_w + 1, size=(args.batch,), dtype=np.int32)
        xw, yw = index_batch(ids, mx.array(starts_w, dtype=mx.int32), Lw)
        g_, l_ = loss_and_grad(params, xw, yw)
        leaves = [v for _, v in tree_flatten((g_, l_))]
        mx.eval(*leaves)

    # ------------------ training loop ------------------
    t0 = time.perf_counter()
    last_print = 0

    def _eval_tree(tree):
        """Materialize a nested tree of MX arrays (if any)."""
        if tree is None:
            return
        leaves = [v for _, v in tree_flatten(tree)]
        if leaves:
            mx.eval(*leaves)

    for step in range(1, args.steps + 1):
        # use precomputed deterministic L and starts
        L = int(L_seq[step - 1])
        starts = mx.array(starts_mat[step - 1], dtype=mx.int32)
        x, y = index_batch(train_ids, starts, L)

        # schedule
        if schedule:
            set_lr(opt, lr_at(step, args.lr, schedule))

        # compute grad + loss at current params
        g, loss_val = loss_and_grad(params, x, y)

        # (NEW) decoupled weight decay: shrink selected params BEFORE the gradient step
        if args.wd > 0:
            ndec = apply_decoupled_weight_decay(
                params,
                getattr(opt, "learning_rate", args.lr),
                args.wd,
                decay_bias=args.wd_bias,
                decay_norm=args.wd_norm,
                decay_embed=args.wd_embed,
            )
            mx.eval(params)
            if step == 1:
                print(f"[wd] Will decay {ndec} tensors per step (AdamW-style).")

        # Keep per-array buckets stable across updates: refresh id->index map *before* the optimizer uses it
        if args.layer_bucket == "per-array":
            rebuild_leaf_map()

        # optimizer update
        if hasattr(opt, "apply_gradients"):
            params = opt.apply_gradients(g, params)
        else:
            out = None
            try:
                out = opt.update(params, g)
            except TypeError:
                out = opt.update(g, params)
            params = params if out is None else out

        # materialize barriers to avoid lazy graph growth
        if args.barrier_every > 0 and (step % args.barrier_every == 0):
            _eval_tree(params)
            if hasattr(opt, "state"):
                _eval_tree(opt.state)

        # periodic eval
        if step == 1 or step - last_print >= args.eval_every:
            last_print = step
            logits = transformer_forward(params, cfg, xe_fixed)
            eval_loss = float(cross_entropy_logits(logits, ye_fixed).item())
            bpc = eval_loss / math.log(2.0)
            elapsed = time.perf_counter() - t0
            print(
                f"[step {step:6d} LR={opt.learning_rate:.4e}] loss={float(loss_val.item()):.4f}  "
                f"eval={eval_loss:.4f}  bpc={bpc:.3f}  L~[{args.lmin},{args.lmax}]  {elapsed/step:.4f}s/step"
            )

    # Final eval
    logits = transformer_forward(params, cfg, xe_fixed)
    loss = float(cross_entropy_logits(logits, ye_fixed).item())
    bpc = loss / math.log(2.0)
    secs = time.perf_counter() - t0
    print(
        f"[step {step:6d} LR={opt.learning_rate:.4e}] loss={float(loss_val.item()):.4f}  "
        f"eval={eval_loss:.4f}  bpc={bpc:.3f}  L~[{args.lmin},{args.lmax}]  {elapsed/step:.4f}s/step"
    )

    print(
        f"[done] {args.steps} steps in {secs:.1f}s  final loss={loss:.4f}  bpc={bpc:.3f}"
    )


if __name__ == "__main__":
    main()
