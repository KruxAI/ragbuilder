"""Load SDK components without initializing unrelated services."""
from importlib import import_module

_EXPORTS = {
    "DBLoggerCallback": "callbacks", "ConfigStore": "config_store",
    "DocumentStore": "document_store", "setup_rich_logging": "logging_utils", "console": "logging_utils",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)
