"""Deprecated path: omicverse ``_single`` was split by plot type.

Prefer ``cellfish.plot._embedding``, ``_composition``, ``_catplot``, ``_contour``.
"""

from ._catplot import (
    bardotplot,
    dotplot_doublegroup,
    plot_boxplots,
    single_group_boxplot,
    violin_box,
    violin_old,
)
from ._composition import cell_alluvial, cellproportion, cellstackarea
from ._contour import add_contour, contour
from ._embedding import (
    ConvexHull,
    add_arrow,
    embedding,
    embedding_adjust,
    embedding_celltype,
    embedding_density,
    mde,
    pca,
    tsne,
    umap,
)

__all__ = [
    "embedding",
    "umap",
    "tsne",
    "pca",
    "mde",
    "embedding_celltype",
    "embedding_adjust",
    "embedding_density",
    "ConvexHull",
    "add_arrow",
    "cellproportion",
    "cellstackarea",
    "cell_alluvial",
    "bardotplot",
    "single_group_boxplot",
    "plot_boxplots",
    "violin_old",
    "violin_box",
    "dotplot_doublegroup",
    "contour",
    "add_contour",
]
