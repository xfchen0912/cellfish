"""Generic palettes. Project-specific colors stay in the analysis repo."""

from __future__ import annotations

from collections.abc import Sequence

from anndata import AnnData
from matplotlib import colormaps


def create_palette_from_types(
    labels: Sequence[str],
    *,
    cmap: str = "tab20",
) -> dict[str, str]:
    """Map unique labels to colors from a matplotlib colormap."""
    unique = list(dict.fromkeys(str(x) for x in labels))
    n = max(len(unique), 1)
    cm = colormaps[cmap]
    colors = [cm(i / max(n - 1, 1)) for i in range(n)]
    return {lab: _to_hex(colors[i]) for i, lab in enumerate(unique)}


def reorder_and_set_palettes(
    adata: AnnData,
    group: str,
    order: Sequence[str] | None = None,
    palette: dict[str, str] | Sequence[str] | None = None,
) -> AnnData:
    """Set categorical order and uns colors for ``adata.obs[group]``."""
    if group not in adata.obs.columns:
        raise KeyError(f"{group!r} not in adata.obs")

    series = adata.obs[group].astype("category")
    if order is not None:
        present = [x for x in order if x in series.cat.categories]
        rest = [x for x in series.cat.categories if x not in present]
        series = series.cat.reorder_categories(present + rest, ordered=True)
        adata.obs[group] = series

    cats = list(adata.obs[group].astype("category").cat.categories)
    if palette is None:
        color_map = create_palette_from_types(cats)
    elif isinstance(palette, dict):
        color_map = palette
    else:
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}

    adata.uns[f"{group}_colors"] = [color_map.get(cat, "#808080") for cat in cats]
    return adata


def _to_hex(color) -> str:
    from matplotlib.colors import to_hex

    return to_hex(color, keep_alpha=False)
