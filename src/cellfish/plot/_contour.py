"""Density contours and obs-column scatter. Not tissue/spatial plotting.

Tissue coordinates should be drawn with ``embedding(..., basis="spatial")``.
"""

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from anndata import AnnData


def add_contour(
    ax,
    adata: AnnData,
    groupby: str,
    clusters: List[str],
    basis: str = "X_umap",
    grid_density: int = 100,
    contour_threshold: float = 0.1,
    **kwargs,
):
    """Add density contour to plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    adata : AnnData
        AnnData object.
    groupby : str
        Column to group by.
    clusters : List[str]
        Clusters to plot.
    basis : str, optional
        Embedding key in adata.obsm.
    grid_density : int, optional
        Grid density for KDE.
    contour_threshold : float, optional
        Threshold for contour levels.
    **kwargs
        Additional arguments for ax.contour.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with contour added.
    """
    umap_embedding = adata[adata.obs[groupby].isin(clusters)].obsm[basis]
    kde = gaussian_kde(umap_embedding.T)

    x_grid = np.linspace(
        min(umap_embedding[:, 0]), max(umap_embedding[:, 0]), grid_density
    )
    y_grid = np.linspace(
        min(umap_embedding[:, 1]), max(umap_embedding[:, 1]), grid_density
    )
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    Z = np.reshape(kde(positions).T, X_grid.shape)

    threshold = np.max(Z) * contour_threshold

    contour = ax.contour(X_grid, Y_grid, Z, levels=[threshold], **kwargs)

    return ax


def plot_scatter(
    adata: AnnData,
    x: str,
    y: str,
    hue: str,
    cmap: Dict[str, str],
    highlight: Optional[Union[str, List[str]]] = None,
    figsize: tuple = (6, 5),
    point_size: int = 5,
    alpha_other: float = 0.5,
    alpha_highlight: float = 0.8,
):
    """Scatter plot from AnnData.obs with optional cluster highlighting.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    hue : str
        Column name to group points by.
    cmap : Dict[str, str]
        Mapping from group name to color.
    highlight : Union[str, List[str]], optional
        Clusters to highlight.
    figsize : tuple, optional
        Figure size.
    point_size : int, optional
        Point size.
    alpha_other : float, optional
        Alpha for background points.
    alpha_highlight : float, optional
        Alpha for highlighted points.
    """
    from matplotlib.patches import Patch

    df = adata.obs.copy()
    df["x"] = df[x]
    df["y"] = df[y]
    df["cluster"] = df[hue].astype(str)

    fig = plt.figure(figsize=figsize)

    if cmap is None:
        cmap = {
            ct: adata.uns[f"{hue}_colors"][i]
            for i, ct in enumerate(adata.obs[hue].cat.categories)
        }

    # Case 1: no highlight → normal scatter by group
    if highlight is None:
        for cl in sorted(df["cluster"].unique()):
            df_cl = df[df["cluster"] == cl]
            plt.scatter(
                df_cl["x"],
                df_cl["y"],
                c=cmap[cl],
                s=point_size,
                alpha=0.5,
                label=f"Cluster {cl}",
            )

        legend_elements = [
            Patch(facecolor=cmap[cl], label=f"Cluster {cl}")
            for cl in sorted(df["cluster"].unique())
        ]

    else:
        # ensure highlight is a list
        if isinstance(highlight, str):
            highlight = [highlight]

        df_highlight = df[df["cluster"].isin(highlight)]
        df_other = df[~df["cluster"].isin(highlight)]

        # plot others (gray)
        plt.scatter(
            df_other["x"],
            df_other["y"],
            c="lightgray",
            s=point_size,
            alpha=alpha_other,
            label="Other clusters",
        )

        # plot highlights
        for cl in highlight:
            df_cl = df_highlight[df_highlight["cluster"] == cl]
            plt.scatter(
                df_cl["x"],
                df_cl["y"],
                c=cmap[cl],
                s=point_size,
                alpha=alpha_highlight,
                label=f"Cluster {cl}",
            )

        legend_elements = [
            Patch(facecolor=cmap[cl], label=f"Cluster {cl}") for cl in highlight
        ]
        legend_elements.append(
            Patch(facecolor="lightgray", label="Other clusters")
        )

    # labels & legend
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(
        handles=legend_elements,
        title=hue,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()


obs_scatter = plot_scatter
contour = add_contour
