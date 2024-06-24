

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = ["run_ragbuilder","__version__"]