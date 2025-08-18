import mlx.core as mx
import mlx.nn as nn

from kbeta import KourkoutasSoftmaxFlex


class Dummy(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.param = mx.array(p)

    def __call__(self):  # not used in the test
        return self.param


def test_smoke_update():
    p = mx.random.uniform(shape=(3,))
    model = Dummy(p)
    grads = {"param": mx.ones_like(p)}  # same tree as model.parameters()

    opt = KourkoutasSoftmaxFlex(learning_rate=1e-3)
    opt.init(model.parameters())
    opt.update(model, grads)  # must not raise
