"""Cell-composition plots: stacked bars, area, alluvial."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from scanpy.plotting._utils import ColorLike

from cellfish.data import require_obs

from ._style import savefig



def get_cluster_proportions(
    adata: AnnData,
    cluster_key: str = "cluster_final",
    sample_key: str = "replicate",
    sort_key: str | None = None,
    drop_values: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return percent composition (0–100) with samples as index and clusters as columns."""
    cols = [cluster_key, sample_key] + ([sort_key] if sort_key else [])
    require_obs(adata, cols)
    obs = adata.obs

    counts = obs.groupby([sample_key, cluster_key], observed=False).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0) * 100

    total_cells = obs[sample_key].value_counts()
    if drop_values is not None:
        props = props.drop(list(drop_values), axis=0, errors="ignore")

    if sort_key is not None:
        sort_order = obs.drop_duplicates(subset=[sample_key, sort_key]).set_index(sample_key)[sort_key]
        props = props.loc[sort_order.index].reindex(sort_order.sort_values().index)

    props.index = [f"{sample} ({int(total_cells.loc[sample])})" for sample in props.index]
    if isinstance(obs[cluster_key].dtype, pd.CategoricalDtype):
        props = props.loc[:, obs[cluster_key].cat.categories]
    return props


def plot_cluster_proportions(
    cluster_props: pd.DataFrame,
    cluster_palette: Sequence[str] | dict[str, str] | None = None,
    xlabel_rotation: int = 0,
    figsize: tuple[float, float] | None = None,
    dpi: float | int | None = None,
    show: bool = True,
    save: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of ``get_cluster_proportions`` output."""
    n_samples = cluster_props.shape[0]
    figure_dpi = float(dpi) if dpi is not None else float(plt.rcParams["figure.dpi"])
    size = figsize if figsize is not None else (max(5, n_samples * 0.5), 5)
    fig, ax = plt.subplots(figsize=size, dpi=figure_dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = None
    plot_df = cluster_props
    if isinstance(cluster_palette, dict):
        colors = [cluster_palette.get(c, "#808080") for c in plot_df.columns]
        cmap = ListedColormap(colors)
    elif cluster_palette is not None:
        cmap = ListedColormap(list(cluster_palette))

    plot_df.plot(kind="bar", stacked=True, ax=ax, legend=False, colormap=cmap)
    ax.grid(False)
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(plot_df.index.name)
    ax.set_ylabel("Proportion")
    fig.tight_layout()
    if save is not None:
        savefig(fig, save)
    if show:
        plt.show()
    return fig


def cellproportion(
    adata: AnnData,
    celltype_clusters: str,
    groupby: str,
    groupby_li=None,
    figsize: tuple = (4, 6),
    ticks_fontsize: int = 12,
    labels_fontsize: int = 12,
    ax=None,
    legend: bool = False,
    legend_awargs=None,
    transpose: bool = False,
    save: str = None,
    **kwargs,
):
    r"""Plot cell proportion of each cell type in each visual cluster.

    Arguments:
        adata: AnnData object.
        celltype_clusters: Cell type clusters.
        groupby: Visual clusters.
        groupby_li: Visual cluster list. (None)
        figsize: Figure size. ((4,6))
        ticks_fontsize: Ticks fontsize. (12)
        labels_fontsize: Labels fontsize. (12)
        ax: Matplotlib axes object. (None)
        legend: Whether to show legend. (False)
        legend_awargs: Legend arguments. ({'ncol':1})
        transpose: Whether to transpose the plot (horizontal bars). (False)

    Returns:
        None

    """

    b = pd.DataFrame(columns=["cell_type", "value", "Week"])
    visual_clusters = groupby
    visual_li = groupby_li
    if visual_li == None:
        adata.obs[visual_clusters] = adata.obs[visual_clusters].astype("category")
        visual_li = adata.obs[visual_clusters].cat.categories

    # Ensure color palettes exist for the requested categories
    import matplotlib

    if f"{celltype_clusters}_colors" not in adata.uns:
        palette = matplotlib.colormaps.get_cmap("tab20")
        colors = [
            matplotlib.colors.to_hex(palette(i % palette.N))
            for i in range(len(adata.obs[celltype_clusters].cat.categories))
        ]
        adata.uns[f"{celltype_clusters}_colors"] = colors
    if f"{visual_clusters}_colors" not in adata.uns:
        palette = matplotlib.colormaps.get_cmap("tab20")
        colors = [
            matplotlib.colors.to_hex(palette(i % palette.N))
            for i in range(len(adata.obs[visual_clusters].cat.categories))
        ]
        adata.uns[f"{visual_clusters}_colors"] = colors

    for i in visual_li:
        b1 = pd.DataFrame()
        test = adata.obs.loc[adata.obs[visual_clusters] == i, celltype_clusters].value_counts()
        b1["cell_type"] = test.index
        b1["value"] = test.values / test.sum()
        b1["Week"] = i.replace("Retinoblastoma_", "")
        b = pd.concat([b, b1])

    plt_data2 = adata.obs[celltype_clusters].value_counts()
    plot_data2_color_dict = dict(
        zip(adata.obs[celltype_clusters].cat.categories, adata.uns["{}_colors".format(celltype_clusters)])
    )
    plt_data3 = adata.obs[visual_clusters].value_counts()
    plot_data3_color_dict = dict(
        zip(
            [i.replace("Retinoblastoma_", "") for i in adata.obs[visual_clusters].cat.categories],
            adata.uns["{}_colors".format(visual_clusters)],
        )
    )
    b["cell_type_color"] = b["cell_type"].map(plot_data2_color_dict)
    b["stage_color"] = b["Week"].map(plot_data3_color_dict)

    if legend_awargs is None:
        legend_awargs = {"ncol": 1}

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    # Use ax to control the image
    # sns.set_theme(style="whitegrid")
    # sns.set_theme(style="ticks")
    n = 0
    all_celltype = adata.obs[celltype_clusters].cat.categories
    for i in all_celltype:
        if n == 0:
            test1 = b[b["cell_type"] == i]
            if transpose:
                ax.barh(
                    y=test1["Week"],
                    width=test1["value"],
                    height=0.8,
                    color=list(set(test1["cell_type_color"]))[0],
                    label=i,
                )
            else:
                ax.bar(
                    x=test1["Week"],
                    height=test1["value"],
                    width=0.8,
                    color=list(set(test1["cell_type_color"]))[0],
                    label=i,
                )
            bottoms = test1["value"].values
        else:
            test2 = b[b["cell_type"] == i]
            if transpose:
                ax.barh(
                    y=test2["Week"],
                    width=test2["value"],
                    left=bottoms,
                    height=0.8,
                    color=list(set(test2["cell_type_color"]))[0],
                    label=i,
                )
            else:
                ax.bar(
                    x=test2["Week"],
                    height=test2["value"],
                    bottom=bottoms,
                    width=0.8,
                    color=list(set(test2["cell_type_color"]))[0],
                    label=i,
                )
            test1 = test2
            bottoms += test1["value"].values
        n += 1
    if legend != False:
        # Merge defaults with user-supplied legend kwargs to avoid duplicate bbox_to_anchor
        legend_kw = {"bbox_to_anchor": (1.05, -0.05), "loc": 3, "borderaxespad": 0, "fontsize": 10}
        legend_kw.update(legend_awargs or {})
        plt.legend(**legend_kw)

    plt.grid(False)

    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Set left and bottom axis ticks to transparent color
    # ax.yaxis.tick_left()
    # ax.xaxis.tick_bottom()
    # ax.xaxis.set_tick_params(color='none')
    # ax.yaxis.set_tick_params(color='none')

    # Set left and bottom axis lines to independent segments
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if transpose:
        plt.yticks(fontsize=ticks_fontsize, rotation=0)
        plt.xticks(fontsize=ticks_fontsize)
        plt.ylabel(groupby, fontsize=labels_fontsize)
        plt.xlabel("Cells per Stage", fontsize=labels_fontsize)
    else:
        plt.xticks(fontsize=ticks_fontsize, rotation=90)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel(groupby, fontsize=labels_fontsize)
        plt.ylabel("Cells per Stage", fontsize=labels_fontsize)
    # fig.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
    if ax == None:
        return fig, ax




def cellstackarea(
    adata,
    celltype_clusters: str,
    groupby: str,
    groupby_li=None,
    figsize: tuple = (4, 6),
    ticks_fontsize: int = 12,
    labels_fontsize: int = 12,
    ax=None,
    legend: bool = False,
    legend_awargs={},
    text_show=False,
):
    """
    Plot the cell type percentage in each groupby category

    """
    df = adata.obs[[groupby, celltype_clusters]]

    # 计算每个样本类型中每个细胞类型的数量
    count_df = df.groupby([groupby, celltype_clusters]).size().reset_index(name="count")

    # 计算每个样本类型中的总数
    total_count_df = count_df.groupby(groupby)["count"].sum().reset_index(name="total_count")

    # 将总数合并回原数据框
    count_df = count_df.merge(total_count_df, on=groupby)

    # 计算百分比
    count_df["percentage"] = count_df["count"] / count_df["total_count"] * 100

    # 将数据从长格式转换为宽格式，以便绘制面积图
    pivot_df = count_df.pivot(index=groupby, columns=celltype_clusters, values="percentage").fillna(0)
    if groupby_li != None:
        pivot_df = pivot_df.loc[groupby_li]

    # 使用 matplotlib 绘制面积图
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    # 为每种细胞类型绘制面积图
    cell_types = pivot_df.columns
    bottom = pd.Series([0] * len(pivot_df), index=pivot_df.index)

    adata.obs[celltype_clusters] = adata.obs[celltype_clusters].astype("category")
    if "{}_colors".format(celltype_clusters) in adata.uns.keys():
        print("{}_colors".format(celltype_clusters))
        type_color_all = dict(
            zip(adata.obs[celltype_clusters].cat.categories, adata.uns["{}_colors".format(celltype_clusters)])
        )
    else:
        if len(adata.obs[celltype_clusters].cat.categories) > 28:
            type_color_all = dict(zip(adata.obs[celltype_clusters].cat.categories, sc.pl.palettes.default_102))
        else:
            type_color_all = dict(zip(adata.obs[celltype_clusters].cat.categories, sc.pl.palettes.zeileis_28))

    for cell_type in cell_types:
        ax.fill_between(
            pivot_df.index, bottom, bottom + pivot_df[cell_type], label=cell_type, color=type_color_all[cell_type]
        )
        max_index = pivot_df[cell_type].idxmax()
        if text_show == True:
            ax.text(
                max_index,
                bottom[max_index] + pivot_df.loc[max_index, cell_type] / 2,
                cell_type,
                fontsize=ticks_fontsize - 1,
            )

        bottom += pivot_df[cell_type]

    if legend != False:
        plt.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0, fontsize=labels_fontsize, **legend_awargs)

    plt.grid(False)

    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Set left and bottom axis ticks to transparent color
    # ax.yaxis.tick_left()
    # ax.xaxis.tick_bottom()
    # ax.xaxis.set_tick_params(color='none')
    # ax.yaxis.set_tick_params(color='none')

    # Set left and bottom axis lines to independent segments
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    plt.xticks(fontsize=ticks_fontsize, rotation=90)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(groupby, fontsize=labels_fontsize)
    plt.ylabel("Cells per Stage", fontsize=labels_fontsize)
    # fig.tight_layout()
    if ax == None:
        return fig, ax


def _celltype_color_map(adata: AnnData, celltype_clusters: str, cell_types: Sequence[str]) -> dict:
    """Resolve colors for *cell_types* from ``adata.uns`` or scanpy defaults."""
    adata.obs[celltype_clusters] = adata.obs[celltype_clusters].astype("category")
    categories = list(adata.obs[celltype_clusters].cat.categories)
    key = f"{celltype_clusters}_colors"
    if key in adata.uns:
        palette = adata.uns[key]
        type_color_all = dict(zip(categories, palette))
    elif len(categories) > 28:
        type_color_all = dict(zip(categories, sc.pl.palettes.default_102))
    else:
        type_color_all = dict(zip(categories, sc.pl.palettes.zeileis_28))
    return {ct: type_color_all[ct] for ct in cell_types if ct in type_color_all}


def _composition_tables(
    adata: AnnData,
    celltype_clusters: str,
    groupby: str,
    *,
    groupby_order: Optional[Sequence[str]] = None,
    celltypes: Optional[Sequence[str]] = None,
    merge_others: bool = False,
    others_label: str = "Other",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return count / percentage pivots and ordered cell-type column names."""
    df = adata.obs[[groupby, celltype_clusters]].copy()
    count_df = df.groupby([groupby, celltype_clusters], observed=True).size().reset_index(name="count")
    total_df = count_df.groupby(groupby, observed=True)["count"].sum().reset_index(name="total_count")
    count_df = count_df.merge(total_df, on=groupby)
    count_df["percentage"] = count_df["count"] / count_df["total_count"] * 100

    count_pivot = count_df.pivot(index=groupby, columns=celltype_clusters, values="count").fillna(0)
    pct_pivot = count_df.pivot(index=groupby, columns=celltype_clusters, values="percentage").fillna(0)

    if groupby_order is not None:
        order = [g for g in groupby_order if g in count_pivot.index]
        count_pivot = count_pivot.loc[order]
        pct_pivot = pct_pivot.loc[order]

    adata.obs[celltype_clusters] = adata.obs[celltype_clusters].astype("category")
    if celltypes is not None:
        selected = [c for c in celltypes if c in count_pivot.columns]
    else:
        selected = [c for c in adata.obs[celltype_clusters].cat.categories if c in count_pivot.columns]

    if merge_others and celltypes is not None:
        other_cols = [c for c in count_pivot.columns if c not in selected]
        if other_cols:
            count_pivot[others_label] = count_pivot[other_cols].sum(axis=1)
            pct_pivot[others_label] = pct_pivot[other_cols].sum(axis=1)
            selected = list(selected) + [others_label]
        count_pivot = count_pivot[selected]
        pct_pivot = pct_pivot[selected]
    else:
        count_pivot = count_pivot[selected]
        pct_pivot = pct_pivot[selected]

    return count_pivot, pct_pivot, list(selected)


def _style_stackarea_ax(ax: Axes, *, panel_facecolor: str = "white") -> None:
    """Match axis styling used in :func:`cellstackarea`."""
    ax.grid(False)
    ax.set_facecolor(panel_facecolor)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


def _stackarea_legend(
    ax: Axes,
    handles: Sequence[mpatches.Patch],
    labels_fontsize: float,
    legend_kw: Mapping[str, Any],
) -> None:
    """Match legend placement used in :func:`cellstackarea`."""
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.05, -0.05),
        loc=3,
        borderaxespad=0,
        fontsize=labels_fontsize,
        **legend_kw,
    )


def _draw_alluvial_panel(
    ax: Axes,
    pivot_df: pd.DataFrame,
    color_map: Mapping[str, ColorLike],
    cell_types: Sequence[str],
    *,
    ylabel: str,
    smooth: Literal["pchip", "linear"] = "pchip",
    show_bars: bool = True,
    bar_width: Optional[float] = None,
    bar_edgewidth: Optional[float] = None,
    ribbon_alpha: float = 1.0,
    n_interp: Optional[int] = None,
    ticks_fontsize: Optional[int] = None,
    labels_fontsize: Optional[int] = None,
    ylim: Optional[Tuple[float, float]] = None,
    panel_facecolor: str = "white",
    xlim_pad: float = 0.35,
) -> None:
    """Draw stacked alluvial ribbons with outlined bars."""
    from scipy.interpolate import PchipInterpolator, interp1d

    if bar_width is None:
        bar_width = 0.28
    if bar_edgewidth is None:
        bar_edgewidth = 1.0
    if n_interp is None:
        n_interp = 120
    if ticks_fontsize is None:
        ticks_fontsize = 12
    if labels_fontsize is None:
        labels_fontsize = 12

    n = len(pivot_df)
    x_pts = np.arange(n, dtype=float)
    x_labels = [str(x) for x in pivot_df.index]

    if n >= 2:
        x_dense = np.linspace(0, n - 1, (n - 1) * n_interp + 1)
    else:
        x_dense = x_pts.copy()

    bottom = np.zeros(n, dtype=float)
    for ct in cell_types:
        if ct not in pivot_df.columns:
            continue

        vals = pivot_df[ct].values.astype(float)
        top = bottom + vals
        color = color_map.get(ct, "#cccccc")

        # 平滑 ribbon
        if n >= 2:
            if smooth == "pchip":
                ib = PchipInterpolator(x_pts, bottom)
                it = PchipInterpolator(x_pts, top)
            else:
                ib = interp1d(x_pts, bottom, kind="linear")
                it = interp1d(x_pts, top, kind="linear")

            ax.fill_between(
                x_dense,
                ib(x_dense),
                it(x_dense),
                color=color,
                alpha=ribbon_alpha,
                linewidth=0,
                zorder=1,
            )

        # 竖直柱子
        if show_bars:
            for i, xi in enumerate(x_pts):
                h = vals[i]
                if h <= 0:
                    continue
                ax.bar(
                    xi,
                    h,
                    bottom=bottom[i],
                    width=bar_width,
                    color=color,
                    edgecolor="black",
                    linewidth=bar_edgewidth,
                    zorder=3,
                    align="center",
                )

        bottom = top

    ax.set_xlim(-xlim_pad, n - 1 + (xlim_pad - 0.15))
    ax.set_xticks(x_pts)
    ax.set_xticklabels(x_labels, fontsize=ticks_fontsize, rotation=90)
    ax.set_ylabel(ylabel, fontsize=labels_fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)

    _style_stackarea_ax(ax, panel_facecolor=panel_facecolor)

def cell_alluvial(
    adata: AnnData,
    celltype_clusters: str,
    groupby: str,
    *,
    groupby_order: Optional[Sequence[str]] = None,
    celltypes: Optional[Sequence[str]] = None,
    merge_others: bool = False,
    others_label: str = "Other",
    mode: Literal["count", "proportion", "both"] = "both",
    smooth: Literal["pchip", "linear"] = "pchip",
    show_bars: bool = True,
    bar_width: Optional[float] = None,
    ribbon_alpha: float = 1.0,
    n_interp: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ticks_fontsize: Optional[int] = None,
    labels_fontsize: Optional[int] = None,
    ax: Optional[Axes] = None,
    legend: bool = True,
    legend_awargs: Optional[dict] = None,
    xlabel: Optional[str] = None,
    figure_facecolor: str = "white",
    panel_facecolor: str = "white",
    title: Optional[str] = None,
):
    """
    Temporal / grouped composition plot with smooth alluvial ribbons.

    Default figure size, typography, axis styling, and legend placement match
    :func:`cellstackarea` (``figsize=(4, 6)``, ``ticks_fontsize=12``,
    ``labels_fontsize=12``, legend at ``bbox_to_anchor=(1.05, -0.05)``).
    For ``mode='both'``, the default width is doubled to ``(8, 6)``.
    """
    if mode == "both" and ax is not None:
        raise ValueError("`ax` cannot be used when mode='both'; pass mode='count' or 'proportion' instead.")

    count_pivot, pct_pivot, cell_types = _composition_tables(
        adata,
        celltype_clusters,
        groupby,
        groupby_order=groupby_order,
        celltypes=celltypes,
        merge_others=merge_others,
        others_label=others_label,
    )
    if not cell_types:
        raise ValueError("No cell types to plot after filtering.")

    if figsize is None:
        figsize = (8, 6) if mode == "both" else (4, 6)

    if bar_width is None:
        bar_width = 0.28
    if n_interp is None:
        n_interp = 120
    if ticks_fontsize is None:
        ticks_fontsize = 12
    if labels_fontsize is None:
        labels_fontsize = 12

    # 配色逻辑保持不变
    color_types = list(cell_types)
    if merge_others and others_label in cell_types:
        color_types = [c for c in color_types if c != others_label] + [others_label]

    color_map = _celltype_color_map(adata, celltype_clusters, color_types)
    if merge_others and others_label in cell_types and others_label not in color_map:
        color_map[others_label] = "#bdbdbd"

    legend_kw = legend_awargs or {}
    xlabel = xlabel if xlabel is not None else groupby

    draw_kw = dict(
        smooth=smooth,
        show_bars=show_bars,
        bar_width=bar_width,
        ribbon_alpha=ribbon_alpha,
        n_interp=n_interp,
        ticks_fontsize=ticks_fontsize,
        labels_fontsize=labels_fontsize,
        panel_facecolor=panel_facecolor,
    )

    handles = [
        mpatches.Patch(facecolor=color_map[ct], label=ct)
        for ct in cell_types
        if ct in color_map
    ]

    if mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
        fig.patch.set_facecolor(figure_facecolor)

        _draw_alluvial_panel(
            axes[0],
            count_pivot,
            color_map,
            cell_types,
            ylabel="Cell number",
            **draw_kw,
        )
        _draw_alluvial_panel(
            axes[1],
            pct_pivot,
            color_map,
            cell_types,
            ylabel="Proportion (%)",
            ylim=(0, 100),
            **draw_kw,
        )

        for a in axes:
            a.set_xlabel(xlabel, fontsize=labels_fontsize)

        if title is not None:
            fig.suptitle(title, fontsize=labels_fontsize)

        if legend and handles:
            _stackarea_legend(axes[1], handles, labels_fontsize, legend_kw)

        return fig, axes

    # 单图模式
    pivot = count_pivot if mode == "count" else pct_pivot
    ylabel = "Cell number" if mode == "count" else "Proportion (%)"
    ylim = None if mode == "count" else (0, 100)

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(figure_facecolor)
    else:
        fig = ax.figure
        fig.patch.set_facecolor(figure_facecolor)

    _draw_alluvial_panel(
        ax,
        pivot,
        color_map,
        cell_types,
        ylabel=ylabel,
        ylim=ylim,
        **draw_kw,
    )
    ax.set_xlabel(xlabel, fontsize=labels_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=labels_fontsize)

    if legend and handles:
        _stackarea_legend(ax, handles, labels_fontsize, legend_kw)

    if created_ax:
        return fig, ax
    return ax



