# src/kbeta/__init__.py
"""
kbeta – Adaptive optimiser family
"""
from importlib.metadata import PackageNotFoundError, version

# Prefer the *package name* here; it's robust in editable & wheel installs.
try:
    __version__ = version("kbeta")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Re-export public API from the subpackage
from .optim import KourkoutasBeta, KourkoutasSoftmaxFlex

__all__ = ["KourkoutasSoftmaxFlex", "KourkoutasBeta", "__version__"]
