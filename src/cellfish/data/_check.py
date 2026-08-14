"""Guards for AnnData fields. Feature names are not assumed to be genes."""

from __future__ import annotations

from collections.abc import Sequence

from anndata import AnnData


def _as_list(names: str | Sequence[str]) -> list[str]:
    if isinstance(names, str):
        return [names]
    return list(names)


def require_obs(adata: AnnData, columns: str | Sequence[str]) -> list[str]:
    """Raise ``KeyError`` if any column is missing from ``adata.obs``."""
    cols = _as_list(columns)
    missing = [c for c in cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing obs columns: {missing}")
    return cols


def require_obsm(adata: AnnData, key: str) -> str:
    """Return the resolved ``obsm`` key, accepting both ``X_umap`` and ``umap``."""
    if key in adata.obsm:
        return key
    alt = key[2:] if key.startswith("X_") else f"X_{key}"
    if alt in adata.obsm:
        return alt
    raise KeyError(f"{key!r} not found in adata.obsm (also tried {alt!r})")


def require_var(adata: AnnData, names: str | Sequence[str]) -> list[str]:
    """Raise ``KeyError`` if any name is missing from ``adata.var_names``."""
    wanted = _as_list(names)
    missing = [n for n in wanted if n not in adata.var_names]
    if missing:
        preview = missing[:5]
        raise KeyError(f"Missing var_names (showing up to 5): {preview}")
    return wanted


def require_layer(adata: AnnData, layer: str | None) -> str | None:
    """Raise ``KeyError`` if ``layer`` is set but missing from ``adata.layers``."""
    if layer is None:
        return None
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} not in adata.layers")
    return layer
