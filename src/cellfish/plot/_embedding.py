"""Embedding scatter wrappers and overlays (omicverse/sc_helpers style)."""

from __future__ import annotations

from typing import (
    Any,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import scanpy as sc

from scanpy.plotting._utils import ColorLike, VBound, _FontSize, _FontWeight

from ._scatterplot import _embedding
from ._scatterplot import embedding_numbered as embedding_numbered

def mde(adata: AnnData, **kwargs):
    r"""
    Plot MDE embedding.

    Arguments:
        adata: Annotated data matrix.
        **kwargs: Additional keyword arguments.

    Returns:
        fig: figure and axis
        ax: axis
    """
    if "X_mde" not in adata.obsm.keys():
        raise ValueError("X_mde not found in adata.obsm. Please run ov.pp.mde first.")
    return embedding(adata, basis="X_mde", **kwargs)


def tsne(adata: AnnData, **kwargs):
    r"""
    Plot t-SNE embedding.

    Arguments:
        adata: Annotated data matrix.
        **kwargs: Additional keyword arguments.

    Returns:
        fig: figure and axis
        ax: axis
    """
    if "X_tsne" not in adata.obsm.keys():
        raise ValueError("X_tsne not found in adata.obsm. Please run ov.pp.tsne first.")
    return embedding(adata, basis="X_tsne", **kwargs)


def pca(adata: AnnData, **kwargs):
    r"""
    Plot PCA embedding.

    Arguments:
        adata: Annotated data matrix.
        **kwargs: Additional keyword arguments.
    """
    if "X_pca" not in adata.obsm.keys() or "scaled|original|X_pca" not in adata.obsm.keys():
        raise ValueError("X_pca not found in adata.obsm. Please run ov.pp.pca first.")
    if "scaled|original|X_pca" in adata.obsm.keys():
        adata.obsm["X_pca"] = adata.obsm["scaled|original|X_pca"]

    return embedding(adata, basis="X_pca", **kwargs)


def umap(adata: AnnData, **kwargs):
    r"""
    Plot UMAP embedding.

    Arguments:
        adata: Annotated data matrix.
        color: Keys for annotations of observations/cells or variables/genes. (None)
        gene_symbols: Key for field in `.var` that stores gene symbols. (None)
        use_raw: Use `.raw` attribute of `adata` if present. (None)
        sort_order: For continuous annotations used as color parameter, plot data points with higher values on top of others. (True)
        edges: Show edges between cells. (False)
        edges_width: Width of edges. (0.1)
        edges_color: Color of edges. ('grey')
        neighbors_key: Key to use for neighbors. (None)
        arrows: Show arrows for velocity. (False)
        arrows_kwds: Keyword arguments for arrow plots. (None)
        groups: Groups to highlight. (None)
        components: Components to plot. (None)
        dimensions: Dimensions to plot. (None)
        layer: Name of the layer to use for coloring. (None)
        projection: Type of projection ('2d' or '3d'). ('2d')
        scale_factor: Scaling factor for sizes. (None)
        color_map: Colormap to use for continuous variables. (None)
        cmap: Colormap to use for continuous variables. (None)
        palette: Colors to use for categorical variables. (None)
        na_color: Color to use for NaN values. ('lightgray')
        na_in_legend: Include NaN values in legend. (True)
        size: Size of the dots. (None)
        frameon: Draw a frame around the plot. ('small')
        legend_fontsize: Font size for legend. (None)
        legend_fontweight: Font weight for legend. ('bold')
        legend_loc: Location of legend. ('right margin')
        legend_fontoutline: Outline width for legend text. (None)
        colorbar_loc: Location of colorbar. ('right')
        vmax: Maximum value for colorbar. (None)
        vmin: Minimum value for colorbar. (None)
        vcenter: Center value for colorbar. (None)
        norm: Normalization for colorbar. (None)
        add_outline: Add outline to points. (False)
        outline_width: Width of outline. ((0.3, 0.05))
        outline_color: Color of outline. (('black', 'white'))
        ncols: Number of columns for subplots. (4)
        hspace: Height spacing between subplots. (0.25)
        wspace: Width spacing between subplots. (None)
        title: Title for the plot. (None)
        show: Show the plot. (None)
        save: Save the plot. (None)
        ax: Matplotlib axes object. (None)
        return_fig: Return figure object. (None)
        marker: Marker style. ('.')
        **kwargs: Additional keyword arguments.

    Returns:
        fig: figure and axis
        ax: axis
    """
    if "X_umap" not in adata.obsm.keys():
        raise ValueError("X_umap not found in adata.obsm. Please run ov.pp.umap first.")
    return embedding(adata, basis="X_umap", **kwargs)


def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = "grey",
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal["2d", "3d"] = "2d",
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = "small",
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    colorbar_width: Optional[float] = None,
    colorbar_pad: Optional[float] = None,
    colorbar_height_fraction: float = 0.3,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = ".",
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""Scatter plot for user specified embedding basis (e.g. umap, pca, etc).

    Arguments:
        adata: Annotated data matrix.
        basis: Name of the `obsm` basis to use.
        color: Keys for annotations of observations/cells or variables/genes. (None)
        gene_symbols: Key for field in `.var` that stores gene symbols. (None)
        use_raw: Use `.raw` attribute of `adata` if present. (None)
        sort_order: For continuous annotations used as color parameter, plot data points with higher values on top of others. (True)
        edges: Show edges between cells. (False)
        edges_width: Width of edges. (0.1)
        edges_color: Color of edges. ('grey')
        neighbors_key: Key to use for neighbors. (None)
        arrows: Show arrows for velocity. (False)
        arrows_kwds: Keyword arguments for arrow plots. (None)
        groups: Groups to highlight. (None)
        components: Components to plot. (None)
        dimensions: Dimensions to plot. (None)
        layer: Name of the layer to use for coloring. (None)
        projection: Type of projection ('2d' or '3d'). ('2d')
        scale_factor: Scaling factor for sizes. (None)
        color_map: Colormap to use for continuous variables. (None)
        cmap: Colormap to use for continuous variables. (None)
        palette: Colors to use for categorical variables. (None)
        na_color: Color to use for NaN values. ('lightgray')
        na_in_legend: Include NaN values in legend. (True)
        size: Size of the dots. (None)
        frameon: Draw a frame around the plot. ('small')
        legend_fontsize: Font size for legend. (None)
        legend_fontweight: Font weight for legend. ('bold')
        legend_loc: Location of legend. ('right margin')
        legend_fontoutline: Outline width for legend text. (None)
        colorbar_loc: Location of colorbar. ('right')
        colorbar_width: Colorbar width as a figure fraction when set; auto-scales
            from panel width when omitted. (None)
        colorbar_pad: Gap between panel and colorbar, as a fraction of panel width. (None)
        colorbar_height_fraction: Colorbar height as a fraction of panel height. (0.3)
        vmax: Maximum value for colorbar. (None)
        vmin: Minimum value for colorbar. (None)
        vcenter: Center value for colorbar. (None)
        norm: Normalization for colorbar. (None)
        add_outline: Add outline to points. (False)
        outline_width: Width of outline. ((0.3, 0.05))
        outline_color: Color of outline. (('black', 'white'))
        ncols: Number of columns for subplots. (4)
        hspace: Height spacing between subplots. (0.25)
        wspace: Width spacing between subplots. (None)
        title: Title for the plot. (None)
        show: Show the plot. (None)
        save: Save the plot. (None)
        ax: Matplotlib axes object. (None)
        return_fig: Return figure object. (None)
        marker: Marker style. ('.')
        **kwargs: Additional keyword arguments.

    Returns:
        ax: If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """

    return _embedding(
        adata=adata,
        basis=basis,
        color=color,
        gene_symbols=gene_symbols,
        use_raw=use_raw,
        sort_order=sort_order,
        edges=edges,
        edges_width=edges_width,
        edges_color=edges_color,
        neighbors_key=neighbors_key,
        arrows=arrows,
        arrows_kwds=arrows_kwds,
        groups=groups,
        components=components,
        dimensions=dimensions,
        layer=layer,
        projection=projection,
        scale_factor=scale_factor,
        color_map=color_map,
        cmap=cmap,
        palette=palette,
        na_color=na_color,
        na_in_legend=na_in_legend,
        size=size,
        frameon=frameon,
        legend_fontsize=legend_fontsize,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        legend_fontoutline=legend_fontoutline,
        colorbar_loc=colorbar_loc,
        colorbar_width=colorbar_width,
        colorbar_pad=colorbar_pad,
        colorbar_height_fraction=colorbar_height_fraction,
        vmax=vmax,
        vmin=vmin,
        vcenter=vcenter,
        norm=norm,
        add_outline=add_outline,
        outline_width=outline_width,
        outline_color=outline_color,
        ncols=ncols,
        hspace=hspace,
        wspace=wspace,
        title=title,
        show=show,
        save=save,
        ax=ax,
        return_fig=return_fig,
        marker=marker,
        **kwargs,
    )



def embedding_celltype(
    adata: AnnData,
    figsize: tuple = (6, 4),
    basis: str = "umap",
    celltype_key: str = "major_celltype",
    title: str = None,
    celltype_range: tuple = (2, 9),
    embedding_range: tuple = (3, 10),
    xlim: int = -1000,
) -> tuple:
    r"""
    Plot embedding with celltype color by omicverse.

    Arguments:
        adata: AnnData object
        figsize: tuple, optional (default=(6,4))
            Figure size
        basis: str, optional (default='umap')
            Embedding method
        celltype_key: str, optional (default='major_celltype')
            Celltype key in adata.obs
        title: str, optional (default=None)
            Figure title
        celltype_range: tuple, optional (default=(2,9))
            Celltype range to plot
        embedding_range: tuple, optional (default=(3,10))
            Embedding range to plot
        xlim: int, optional (default=-1000)
            X axis limit

    Returns:
        fig: figure and axis
        ax: axis
    """

    adata.obs[celltype_key] = adata.obs[celltype_key].astype("category")
    cell_counts = adata.obs[celltype_key].value_counts()
    cell_num_pd = pd.DataFrame({celltype_key: cell_counts.values}, index=cell_counts.index)

    if "{}_colors".format(celltype_key) in adata.uns.keys():
        cell_color_dict = dict(
            zip(adata.obs[celltype_key].cat.categories.tolist(), adata.uns["{}_colors".format(celltype_key)])
        )
    else:
        if len(adata.obs[celltype_key].cat.categories) > 28:
            cell_color_dict = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.default_102))
        else:
            cell_color_dict = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.zeileis_28))

    if figsize == None:
        if len(adata.obs[celltype_key].cat.categories) < 10:
            fig = plt.figure(figsize=(6, 4))
        else:
            print("The number of cell types is too large, please set the figsize parameter")
            return
    else:
        fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(10, 10)
    ax1 = fig.add_subplot(grid[:, embedding_range[0] : embedding_range[1]])  # 占据第一行的所有列
    ax2 = fig.add_subplot(grid[celltype_range[0] : celltype_range[1], :2])
    # Define subplot size and position
    # Occupy the first two columns of the second row
    # ax3 = fig.add_subplot(grid[1:, 2])      # Occupy the last column of the second row and later
    # ax4 = fig.add_subplot(grid[2, 0])       # Occupy the first column of the last row
    # ax5 = fig.add_subplot(grid[2, 1])       # Occupy the second column of the last row

    sc.pl.embedding(
        adata,
        basis=basis,
        color=[celltype_key],
        title="",
        frameon=False,
        # wspace=0.65,
        ncols=3,
        ax=ax1,
        legend_loc=False,
        show=False,
    )

    for idx, cell in zip(range(cell_num_pd.shape[0]), adata.obs[celltype_key].cat.categories):
        ax2.scatter(100, cell, c=cell_color_dict[cell], s=50)
        ax2.plot((100, cell_num_pd.loc[cell, celltype_key]), (idx, idx), c=cell_color_dict[cell], lw=4)
        ax2.text(
            100, idx + 0.2, cell + "(" + str("{:,}".format(cell_num_pd.loc[cell, celltype_key])) + ")", fontsize=11
        )
    ax2.set_xlim(xlim, cell_num_pd.iloc[1].values[0])
    ax2.text(xlim, idx + 1, title, fontsize=12)
    ax2.grid(False)
    # ax2.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0,fontsize=10,**legend_awargs)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.axis("off")

    # ——关键：确保 ax2 没有图例——
    if ax1.get_legend() is not None:  # 如果有，就移除
        ax1.get_legend().remove()
    if ax2.get_legend() is not None:  # 如果有，就移除
        ax2.get_legend().remove()

    return fig, [ax1, ax2]


def ConvexHull(adata: AnnData, basis: str, cluster_key: str, hull_cluster: str, ax, color=None, alpha: float = 0.2):
    r"""
    Plot the ConvexHull for a cluster in embedding.

    Arguments:
        adata: AnnData object
        basis: str
            Embedding method in adata.obsm
        cluster_key: str
            Cluster key in adata.obs
        hull_cluster: str
            Cluster to plot for ConvexHull
        ax: matplotlib.axes.Axes
            Axes object
        color: str, optional (default=None)
            Color for ConvexHull
        alpha: float, optional (default=0.2)
            Alpha for ConvexHull

    Returns:
        ax: matplotlib.axes.Axes
            Modified axes object
    """
    from scipy.spatial import ConvexHull

    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    if "{}_colors".format(cluster_key) in adata.uns.keys():
        print("{}_colors".format(cluster_key))
        type_color_all = dict(zip(adata.obs[cluster_key].cat.categories, adata.uns["{}_colors".format(cluster_key)]))
    else:
        if len(adata.obs[cluster_key].cat.categories) > 28:
            type_color_all = dict(zip(adata.obs[cluster_key].cat.categories, sc.pl.palettes.default_102))
        else:
            type_color_all = dict(zip(adata.obs[cluster_key].cat.categories, sc.pl.palettes.zeileis_28))

    # color_dict=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns[f'{cluster_key}_colors']))
    points = adata[adata.obs[cluster_key] == hull_cluster].obsm[basis]
    hull = ConvexHull(points)
    vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
    if color == None:
        ax.plot(points[vert, 0], points[vert, 1], "--", c=type_color_all[hull_cluster])
        ax.fill(points[vert, 0], points[vert, 1], c=type_color_all[hull_cluster], alpha=alpha)
    else:
        ax.plot(points[vert, 0], points[vert, 1], "--", c=color)
        ax.fill(points[vert, 0], points[vert, 1], c=color, alpha=alpha)
    return ax


def embedding_adjust(adata, groupby, exclude=(), basis="X_umap", ax=None, adjust_kwargs=None, text_kwargs=None):
    r"""
    Get locations of cluster median and adjust text labels accordingly.

    Borrowed from scanpy github forum.

    Arguments:
        adata: AnnData object
        groupby: str
            Key in adata.obs for grouping
        exclude: tuple, optional (default=())
            Groups to exclude from labeling
        basis: str, optional (default='X_umap')
            Embedding basis key in adata.obsm
        ax: matplotlib.axes.Axes, optional (default=None)
            Axes object to plot on
        adjust_kwargs: dict, optional (default=None)
            Arguments for adjust_text function
        text_kwargs: dict, optional (default=None)
            Arguments for text annotation

    Returns:
        medians: dict
            Dictionary of median positions for each group
    """
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm[basis], axis=0)

    if ax is None:
        texts = [plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]
    from adjustText import adjust_text

    adjust_text(texts, **adjust_kwargs)
    return texts


def embedding_density(adata, basis, groupby, target_clusters, **kwargs):
    if "X_" in basis:
        basis1 = basis.split("_")[1]
    sc.tl.embedding_density(adata, basis=basis1, groupby=groupby, key_added="temp_density")
    adata.obs.loc[adata.obs[groupby] != target_clusters, "temp_density"] = 0
    return embedding(adata, basis=basis, color=["temp_density"], title=target_clusters, **kwargs)




def add_arrow(ax, adata, basis, fontsize=12, x_label=None, y_label=None, arrow_scale=5, arrow_width=0.01):
    r"""
    Add arrow and label to the axis
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add the arrow and label to.
    adata : AnnData
        The AnnData object to add the arrow and label to.
    basis : str
        The basis to add the arrow and label to.
    fontsize : int
        The fontsize of the label.
    arrow_scale : float
        The scale of the arrow.
    arrow_width : float
        The width of the arrow.
    x_label : str
        The label of the x-axis.
    y_label : str
        The label of the y-axis.
    """
    # Resolve obsm key and pretty labels: X_umap → "UMAP 1" / "UMAP 2"
    if basis in adata.obsm:
        obsm_key = basis
    elif f"X_{basis}" in adata.obsm:
        obsm_key = f"X_{basis}"
    else:
        raise KeyError(f"Could not find '{basis}' or 'X_{basis}' in .obsm")

    key_name = obsm_key[2:] if obsm_key.startswith("X_") else obsm_key
    pretty = {
        "umap": "UMAP",
        "tsne": "t-SNE",
        "pca": "PC",
        "diffmap": "DC",
    }.get(
        key_name.lower(),
        key_name.replace("draw_graph_", "").upper()
        if "draw_graph" in key_name
        else (key_name.upper() if key_name.islower() else key_name),
    )
    if x_label is None:
        x_label = f"{pretty} 1"
    if y_label is None:
        y_label = f"{pretty} 2"

    coords = np.asarray(adata.obsm[obsm_key])
    x_range = (coords[:, 0].max() - coords[:, 0].min()) / 6
    y_range = (coords[:, 1].max() - coords[:, 1].min()) / 6
    x_min = float(coords[:, 0].min())
    y_min = float(coords[:, 1].min())

    # Both arrows start at the corner so tails do not protrude past the origin.
    ax.arrow(
        x=x_min,
        y=y_min,
        dx=x_range + x_range / arrow_scale,
        dy=0,
        width=arrow_width,
        color="k",
        head_width=y_range * 2 / arrow_scale,
        head_length=x_range * 2 / arrow_scale,
        overhang=0,
        length_includes_head=True,
    )

    ax.arrow(
        x=x_min,
        y=y_min,
        dx=0,
        dy=y_range + y_range / arrow_scale,
        width=arrow_width,
        color="k",
        head_width=x_range * 2 / arrow_scale,
        head_length=y_range * 2 / arrow_scale,
        overhang=0,
        length_includes_head=True,
    )
    ax.text(
        x=x_min + x_range * 0.55,
        y=y_min - y_range / 2.5,
        s=x_label,
        fontsize=fontsize,
        multialignment="center",
        verticalalignment="center",
        horizontalalignment="center",
    )
    ax.text(
        x=x_min - x_range / 2.5,
        y=y_min + y_range * 0.55,
        s=y_label,
        fontsize=fontsize,
        rotation="vertical",
        multialignment="center",
        horizontalalignment="center",
        verticalalignment="center",
    )

