from .core.builder import RAGBuilder

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"

__all__ = ['RAGBuilder', '__version__']
