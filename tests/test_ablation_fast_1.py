# tests/test_ablation_fast.py
import mlx.core as mx
from mlx.optimizers import Adam as MLXAdam

from kbeta.optim import KourkoutasSoftmaxFlex

# ----- shared toy setup (pure dict-of-arrays; no Module) ---------------------

# fixed feature vector for a tiny quadratic loss
X = mx.arange(4.0, dtype=mx.float32)


def grad_w(w):
    """
    Gradient for L(w) = 0.5 * ( <w, X> )^2
    dL/dw = (<w, X>) * X
    """
    y = mx.sum(w * X)
    return y * X


# ----- tests -----------------------------------------------------------------


def test_per_step_equivalence_kbeta_as_adam():
    """
    KourkoutasSoftmaxFlex configured as Adam must track MLX Adam
    *step-by-step* on the same toy problem.
    """

    # initial weights (non-zero so grads aren't trivial)
    w0 = mx.arange(1.0, 5.0, dtype=mx.float32)

    # MLX Adam (bias correction ON to match our 'beta2max' mode)
    adam = MLXAdam(
        learning_rate=1e-3,
        betas=[0.9, 0.999],
        eps=1e-8,
        bias_correction=True,
    )
    p_adam = {"w": w0 + 0}
    adam.init(p_adam)

    # K-β configured as Adam (fixed β2, no extras)
    kbeta_as_adam = KourkoutasSoftmaxFlex(
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
        layer_key_fn=lambda _: 0,
        diagnostics=False,
    )
    p_kb = {"w": w0 + 0}
    # Optional explicit init; apply_gradients would also lazily init
    kbeta_as_adam.init(p_kb)

    # A short loop is enough to catch divergences; keep it fast for CI
    for t in range(64):
        g_a = {"w": grad_w(p_adam["w"])}
        g_k = {"w": grad_w(p_kb["w"])}  # identical as long as params match

        adam.update(p_adam, g_a)  # MLX mutates in-place
        p_kb = kbeta_as_adam.apply_gradients(g_k, p_kb)  # returns new params

        # force evaluation (MLX is lazy)
        mx.eval(p_adam["w"], p_kb["w"])

        ok = mx.allclose(p_adam["w"], p_kb["w"], rtol=1e-7, atol=1e-6).item()
        assert ok, f"mismatch at step {t}"


def test_dynamic_beta2_stays_within_bounds():
    """
    With dynamic β2 enabled, the last-per-layer β2 reported via diagnostics
    must lie within [β2_min, β2_max].
    """
    w0 = mx.arange(1.0, 5.0, dtype=mx.float32)
    opt = KourkoutasSoftmaxFlex(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_min=0.88,
        beta2_max=0.999,
        eps=1e-8,
        alpha=0.93,
        decay=None,
        max_ratio=None,
        adaptive_tiny=False,
        warmup_steps=5,
        bias_correction="beta2max",
        layer_key_fn=lambda _: 0,  # single bucket
        diagnostics=True,
    )
    params = {"w": w0 + 0}
    opt.init(params)

    for _ in range(32):
        grads = {"w": grad_w(params["w"])}
        params = opt.apply_gradients(grads, params)
        mx.eval(params["w"])

    spikes, betas = opt.snapshot_sunspike_history()
    assert betas, "No β2 history collected (diagnostics=True?)"
    assert all(0.88 <= b <= 0.999 for b in betas)


# helper: collect per-parameter state dicts (those that contain 'm' and 'v')
def _gather_param_states(state_tree):
    found = []

    def visit(node):
        if isinstance(node, dict):
            if "m" in node and "v" in node:
                found.append(node)
            else:
                for v in node.values():
                    visit(v)

    visit(state_tree)
    return found


def test_trust_region_implies_vmax_alloc():
    """
    Turning on max_ratio should allocate v_max buffers (AMSGrad path).
    """
    w0 = mx.arange(1.0, 5.0, dtype=mx.float32)

    # (1) No trust-region -> no v_max
    opt1 = KourkoutasSoftmaxFlex(
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
    p1 = {"w": w0 + 0}
    opt1.init(p1)
    states1 = _gather_param_states(opt1.state)
    assert states1 and all("v_max" not in st for st in states1)

    # (2) Trust-region ON -> v_max present
    opt2 = KourkoutasSoftmaxFlex(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_min=0.999,
        beta2_max=0.999,
        eps=1e-8,
        alpha=0.9,
        decay=None,
        adaptive_tiny=False,
        max_ratio=3.0,
        bias_correction="none",
        warmup_steps=0,
        layer_key_fn=lambda _: 0,
        diagnostics=False,
    )
    p2 = {"w": w0 + 0}
    opt2.init(p2)
    states2 = _gather_param_states(opt2.state)
    assert any("v_max" in st for st in states2)
