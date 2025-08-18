# tests/test_ablation_fast.py
import mlx.core as mx
from mlx.optimizers import Adam as MLXAdam

from kbeta import KourkoutasBeta  # your class

# --- tiny quadratic: L(w) = (x·w)^2 ----------------------------------------
x = mx.arange(4.0)


def _make_grad_w(x_const):
    # loss: scalar -> OK for mx.grad
    def loss(w):
        return mx.sum(w * x_const) ** 2

    return mx.grad(loss)


grad_w = _make_grad_w(x)


def _gather_param_states(state_tree):
    """Collect per-parameter state dicts (those that contain 'm' and 'v')."""
    found = []

    def visit(node):
        if isinstance(node, dict):
            if "m" in node and "v" in node:  # leaf of a param
                found.append(node)
            else:
                for v in node.values():
                    visit(v)

    visit(state_tree)
    return found


def test_kbeta_as_adam_matches_mlx_adam_per_step():
    """
    Configure Kourkoutas-β to be *exactly* Adam (fixed β2, no extras) and
    assert *per-step* equality with MLX Adam on the same scalar loss.
    """
    # initial parameter (vector) as a raw dict (no nn.Module)
    w0 = mx.ones((4,))
    p_adam = {"w": w0 + 0}
    p_kb = {"w": w0 + 0}

    # MLX Adam - match your ablation script signature
    adam = MLXAdam(
        learning_rate=1e-3,
        betas=[0.9, 0.999],
        eps=1e-8,
        bias_correction=True,
    )
    adam.init(p_adam)

    # Kβ configured as Adam (fixed β2, no extras)
    kbeta_as_adam = KourkoutasBeta(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_max=0.999,
        beta2_min=0.999,  # fixed β2
        eps=1e-8,
        alpha=0.95,
        decay=None,
        adaptive_tiny=False,
        max_ratio=None,
        bias_correction="beta2max",
        warmup_steps=0,
        layer_key_fn=lambda _: 0,  # single bucket
        diagnostics=False,
    )
    kbeta_as_adam.init(p_kb)

    # fewer steps than the manuscript to keep CI fast
    for t in range(50):
        g_a = {"w": grad_w(p_adam["w"])}
        g_k = {"w": grad_w(p_kb["w"])}  # grads identical if params match

        # step both optimizers
        adam.update(p_adam, g_a)  # MLX Adam mutates in-place
        p_kb = kbeta_as_adam.apply_gradients(g_k, p_kb)  # returns new dict

        # force compute & check equality this *step*
        mx.eval(p_adam["w"], p_kb["w"])
        assert mx.allclose(
            p_adam["w"], p_kb["w"], rtol=1e-7, atol=1e-6
        ), f"mismatch at step {t}"


def test_dynamic_beta2_stays_within_bounds():
    """
    With diagnostics enabled, the last β2 value recorded per bucket must lie
    within [β2_min, β2_max].
    """
    p = {"w": mx.ones((4,))}
    opt = KourkoutasBeta(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_min=0.88,
        beta2_max=0.999,
        eps=1e-8,
        alpha=0.93,
        decay=None,
        adaptive_tiny=False,
        max_ratio=None,
        warmup_steps=5,
        bias_correction="beta2max",
        layer_key_fn=lambda _: 0,  # single bucket
        diagnostics=True,  # enables snapshot API
    )
    opt.init(p)

    for _ in range(20):
        g = {"w": grad_w(p["w"])}
        p = opt.apply_gradients(g, p)

    spikes, betas = opt.snapshot_sunspike_history()
    assert betas, "No β2 history collected (is diagnostics=True?)"
    assert all(0.88 <= b <= 0.999 for b in betas)


def test_trust_region_implies_vmax_alloc():
    """
    Enabling max_ratio should allocate v_max buffers (uses AMSGrad path).
    """
    p1 = {"w": mx.ones((4,))}
    opt1 = KourkoutasBeta(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_min=0.999,
        beta2_max=0.999,
        eps=1e-8,
        alpha=0.9,
        decay=None,
        adaptive_tiny=False,
        max_ratio=None,
        bias_correction="none",
        warmup_steps=0,
        layer_key_fn=lambda _: 0,
        diagnostics=False,
    )
    opt1.init(p1)
    states1 = _gather_param_states(opt1.state)
    assert states1 and all("v_max" not in st for st in states1)

    p2 = {"w": mx.ones((4,))}
    opt2 = KourkoutasBeta(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_min=0.999,
        beta2_max=0.999,
        eps=1e-8,
        alpha=0.9,
        decay=None,
        adaptive_tiny=False,
        max_ratio=3.0,  # trust-region ON
        bias_correction="none",
        warmup_steps=0,
        layer_key_fn=lambda _: 0,
        diagnostics=False,
    )
    opt2.init(p2)
    states2 = _gather_param_states(opt2.state)
    assert any("v_max" in st for st in states2)
