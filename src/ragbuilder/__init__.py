"""RAGBuilder's public SDK, loaded only when requested."""
import os

# Keep third-party analytics separate from RAGBuilder's documented usage events.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["RAGBuilder", "__version__"]


def __getattr__(name):
    if name == "RAGBuilder":
        from .core.builder import RAGBuilder
        return RAGBuilder
    raise AttributeError(name)
