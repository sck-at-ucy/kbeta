#  ──  src/kbeta/__init__.py  ──
"""
kbeta – Adaptive optimiser family
"""

from importlib.metadata import PackageNotFoundError, version

try:  # runtime lookup works in editable & wheel installs
    __version__ = version(__name__)
except PackageNotFoundError:  # during CI sdist build
    __version__ = "0.0.0.dev0"

from .optim.kbeta_softmax import KourkoutasSoftmaxFlex

__all__ = ["KourkoutasSoftmaxFlex"]
