# tests/test_semantics.py
import mlx.core as mx
import mlx.nn as nn
import pytest

from kbeta.optim import KourkoutasSoftmaxFlex  # <-- this is the class you have


# ----- tiny toy model -------------------------------------------------------
def tiny_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            # Register as a trainable parameter
            self.w = mx.ones((4,))

        def __call__(self, x):
            # simple linear map -> scalar
            return (self.w * x).sum()

    return M()


def adam_like_kwargs(beta2=0.999, bc="none"):
    """Configure Kourkoutas-β to behave like Adam (fixed β2, no extras)."""
    return dict(
        learning_rate=1e-3,
        beta1=0.9,
        beta2_min=beta2,
        beta2_max=beta2,
        eps=1e-8,
        alpha=0.9,  # irrelevant when β2 is fixed
        decay=None,
        max_ratio=None,
        adaptive_tiny=False,
        warmup_steps=0,
        bias_correction=bc,  # "none" | "beta2max"
        layer_key_fn=lambda _: 0,  # single global bucket
        diagnostics=False,
    )


# ----- tests ----------------------------------------------------------------


@pytest.mark.parametrize("bc", ["none", "beta2max"])
def _gather_param_states(state_tree):
    """Collect only the per-parameter state dicts (those that contain 'm')."""
    found = []

    def visit(node):
        if isinstance(node, dict):
            # per-parameter leaf dicts have 'm' (first moment)
            if "m" in node and "v" in node:
                found.append(node)
            else:
                for v in node.values():
                    visit(v)

    visit(state_tree)
    return found


def test_trust_region_implies_vmax_alloc():
    """Enabling max_ratio should allocate v_max buffers (AMSGrad path)."""
    # no trust region -> no v_max in any per-parameter state
    m1 = tiny_model()
    opt1 = KourkoutasSoftmaxFlex(**adam_like_kwargs())
    opt1.init(m1.parameters())
    states1 = _gather_param_states(opt1.state)
    assert states1 and all("v_max" not in st for st in states1)

    # trust region on -> v_max present
    m2 = tiny_model()
    kwargs = adam_like_kwargs()
    kwargs["max_ratio"] = 3.0
    opt2 = KourkoutasSoftmaxFlex(**kwargs)
    opt2.init(m2.parameters())
    states2 = _gather_param_states(opt2.state)
    assert any("v_max" in st for st in states2)
