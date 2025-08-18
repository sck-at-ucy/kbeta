# src/kbeta/optim/__init__.py
from .kbeta_softmax import KourkoutasSoftmaxFlex

# Back-compat alias
KourkoutasBeta = KourkoutasSoftmaxFlex

__all__ = ["KourkoutasSoftmaxFlex", "KourkoutasBeta"]
