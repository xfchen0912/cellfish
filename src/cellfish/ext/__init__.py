"""Algorithm extras. Imported lazily so ``import cellfish`` stays light."""

from importlib import import_module

_TOOLS = ("archr", "chrombpnet", "drvi", "liana", "milo", "scenicplus")


def __getattr__(name: str):
    if name in _TOOLS:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_TOOLS))
