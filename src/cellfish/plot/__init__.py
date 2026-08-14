"""Visualization utilities. Layer-1 plots only; algorithm plots live in ``cellfish.ext``."""

from ._catplot import (
    bardotplot,
    dotplot_doublegroup,
    plot_boxplots,
    single_group_boxplot,
    violin_box,
    violin_old,
)
from ._composition import (
    cell_alluvial,
    cellproportion,
    cellstackarea,
    get_cluster_proportions,
    plot_cluster_proportions,
)
from ._contour import add_contour, contour, obs_scatter, plot_scatter
from ._embedding import (
    ConvexHull,
    add_arrow,
    embedding,
    embedding_adjust,
    embedding_celltype,
    embedding_density,
    embedding_numbered,
    mde,
    pca,
    tsne,
    umap,
)
from ._fonts import export_mplstyle, font_signature, validate_and_load_fonts
from ._palettes import (
    blue_color,
    cet_g_bw,
    create_palette_from_types,
    get_palette,
    green_color,
    list_available_palettes,
    orange_color,
    order_labels,
    palette_28,
    palette_56,
    palette_112,
    purple_color,
    red_color,
    reorder_and_set_palettes,
    sc_color,
    show_color,
    show_palette,
)
from ._plot1cell import plot1cell
from ._plot1cell_atlas import plot1cell_atlas_meta_rings, simulate_atlas_anndata
from ._ridgeplot import ridgeplot
from ._style import savefig, setup_style

_MARSILEA_EXPORTS = {
    "CircleLabels": ("._grid_dotplot", "CircleLabels"),
    "dotplot": ("._dotplot", "dotplot"),
    "grid_dotplot": ("._grid_dotplot", "grid_dotplot"),
    "rank_genes_groups_dotplot": ("._dotplot", "rank_genes_groups_dotplot"),
}


def __getattr__(name: str):
    if name in _MARSILEA_EXPORTS:
        mod_name, attr = _MARSILEA_EXPORTS[name]
        try:
            module = __import__(f"{__name__}{mod_name}", fromlist=[attr])
        except ImportError as exc:
            raise ImportError(
                f"{name} requires marsilea. Install with: pip install cellfish[plot]"
            ) from exc
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_MARSILEA_EXPORTS))


__all__ = [
    "setup_style",
    "savefig",
    "font_signature",
    "validate_and_load_fonts",
    "export_mplstyle",
    "sc_color",
    "red_color",
    "green_color",
    "orange_color",
    "blue_color",
    "purple_color",
    "palette_28",
    "palette_56",
    "palette_112",
    "cet_g_bw",
    "get_palette",
    "create_palette_from_types",
    "order_labels",
    "reorder_and_set_palettes",
    "show_color",
    "show_palette",
    "list_available_palettes",
    "embedding",
    "embedding_numbered",
    "umap",
    "tsne",
    "pca",
    "mde",
    "embedding_celltype",
    "embedding_adjust",
    "embedding_density",
    "ConvexHull",
    "add_arrow",
    "bardotplot",
    "cellproportion",
    "cellstackarea",
    "cell_alluvial",
    "contour",
    "dotplot_doublegroup",
    "plot_boxplots",
    "single_group_boxplot",
    "violin_box",
    "violin_old",
    "add_contour",
    "plot_scatter",
    "obs_scatter",
    "ridgeplot",
    "get_cluster_proportions",
    "plot_cluster_proportions",
    "plot1cell",
    "plot1cell_atlas_meta_rings",
    "simulate_atlas_anndata",
    "dotplot",
    "grid_dotplot",
    "CircleLabels",
    "rank_genes_groups_dotplot",
]
