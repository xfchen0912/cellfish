"""Marsilea-based grid dotplot with grouped axes and colored labels."""

from __future__ import annotations

from itertools import groupby
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import marsilea as ma
import marsilea.plotter as mp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from marsilea.plotter.base import RenderPlan
from ._dotplot import _marsilea_figure
from matplotlib.colors import Colormap

_VarNames = Union[str, Sequence[str]]


def _get_label_colors(
    adata: AnnData,
    items: Sequence[str],
    obs_key: Optional[str] = None,
) -> Dict[str, str]:
    """Extract colors from ``adata.uns`` or fall back to tab20."""
    palette: Dict[str, str] = {}

    if obs_key is not None and f"{obs_key}_colors" in adata.uns:
        if pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
            cats_in_obs = list(adata.obs[obs_key].cat.categories)
        else:
            cats_in_obs = list(adata.obs[obs_key].unique())

        uns_colors = adata.uns[f"{obs_key}_colors"]
        for i, cat in enumerate(cats_in_obs):
            if i < len(uns_colors):
                palette[cat] = uns_colors[i]

    cmap = plt.get_cmap("tab20")
    default_colors = cmap.colors
    missing = [x for x in items if x not in palette]
    for i, item in enumerate(missing):
        palette[item] = mcolors.to_hex(default_colors[i % len(default_colors)])

    return palette


class CircleLabels(RenderPlan):
    """Draw colored dots with text labels beside them (Marsilea RenderPlan)."""

    def __init__(
        self,
        labels,
        split_groups=None,
        palette=None,
        dot_size=100,
        fontsize=12,
        text_color="black",
        spacing=2,
        **kwargs,
    ):
        self.labels = np.asarray(labels)
        self.split_groups = np.asarray(split_groups) if split_groups is not None else None
        self.palette = palette or {}
        self.dot_size = dot_size
        self.fontsize = fontsize
        self.text_color = text_color
        self.spacing = spacing
        super().__init__(**kwargs)

    def render(self, ax):
        axes = np.atleast_1d(ax)

        if self.split_groups is not None and len(self.split_groups) > 0:
            chunk_sizes = [len(list(g)) for _, g in groupby(self.split_groups)]
        else:
            chunk_sizes = [len(self.labels)]

        idx = 0
        for i, a in enumerate(axes):
            size = chunk_sizes[i]
            chunk_labels = self.labels[idx : idx + size]
            idx += size

            colors = [self.palette.get(lab, "gray") for lab in chunk_labels]
            coords = np.arange(len(chunk_labels))

            if self.side == "left":
                a.scatter(np.zeros_like(coords), coords, c=colors, s=self.dot_size, clip_on=False)
                for y, text in zip(coords, chunk_labels):
                    a.text(
                        -self.spacing,
                        y,
                        str(text),
                        ha="right",
                        va="center",
                        color=self.text_color,
                        fontsize=self.fontsize,
                        clip_on=False,
                    )
                a.set_xlim(-self.spacing - 0.5, 0.5)
                a.set_ylim(len(coords) - 0.5, -0.5)

            elif self.side == "right":
                a.scatter(np.zeros_like(coords), coords, c=colors, s=self.dot_size, clip_on=False)
                for y, text in zip(coords, chunk_labels):
                    a.text(
                        self.spacing,
                        y,
                        str(text),
                        ha="left",
                        va="center",
                        color=self.text_color,
                        fontsize=self.fontsize,
                        clip_on=False,
                    )
                a.set_xlim(-0.5, self.spacing + 0.5)
                a.set_ylim(len(coords) - 0.5, -0.5)

            elif self.side == "bottom":
                a.scatter(coords, np.zeros_like(coords), c=colors, s=self.dot_size, clip_on=False)
                for x, text in zip(coords, chunk_labels):
                    a.text(
                        x,
                        -self.spacing,
                        str(text),
                        ha="center",
                        va="top",
                        rotation=90,
                        color=self.text_color,
                        fontsize=self.fontsize,
                        clip_on=False,
                    )
                a.set_xlim(-0.5, len(coords) - 0.5)
                a.set_ylim(-self.spacing - 0.5, 0.5)

            elif self.side == "top":
                a.scatter(coords, np.zeros_like(coords), c=colors, s=self.dot_size, clip_on=False)
                for x, text in zip(coords, chunk_labels):
                    a.text(
                        x,
                        self.spacing,
                        str(text),
                        ha="center",
                        va="bottom",
                        rotation=90,
                        color=self.text_color,
                        fontsize=self.fontsize,
                        clip_on=False,
                    )
                a.set_xlim(-0.5, len(coords) - 0.5)
                a.set_ylim(-0.5, self.spacing + 0.5)

            a.set_axis_off()


def grid_dotplot(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: str,
    *,
    categories_order: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    swap_axes: bool = False,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Optional[Literal["var", "group"]] = None,
    cmap: Union[Colormap, str, None] = "Reds",
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: int = 12,
    colorbar_title: Optional[str] = "Mean expression",
    size_title: Optional[str] = "Fraction of cells\nin group (%)",
    x_label_colors: Optional[Dict[str, str]] = None,
    y_label_colors: Optional[Dict[str, str]] = None,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = False,
    **kwds,
):
    """Grouped dotplot with Marsilea, supporting 2D category splits and colored labels.

    Parameters
    ----------
    adata
        AnnData with expression values.
    var_names
        Genes / features to plot. A mapping creates column (or row) groups.
    groupby
        Observation column used for grouping.
    categories_order
        Order of groups. A mapping creates row (or column) groups.
    swap_axes
        If True, swap rows and columns (genes on Y, groups on X).
    """
    # 1. Parse var_names
    var_groups: list = []
    var_names_list: list = []
    if isinstance(var_names, Mapping):
        for group, genes in var_names.items():
            if isinstance(genes, str):
                genes = [genes]
            var_names_list.extend(genes)
            var_groups.extend([group] * len(genes))
    else:
        var_names_list = [var_names] if isinstance(var_names, str) else list(var_names)

    # 2. Parse categories_order
    obs_groups: list = []
    cats: list = []
    if isinstance(categories_order, Mapping):
        for group, items in categories_order.items():
            if isinstance(items, str):
                items = [items]
            cats.extend(items)
            obs_groups.extend([group] * len(items))
    else:
        if categories_order is not None:
            cats = list(categories_order)
        elif pd.api.types.is_categorical_dtype(adata.obs[groupby]):
            cats = list(adata.obs[groupby].cat.categories)
        else:
            cats = list(adata.obs[groupby].unique())

    # 3. Expression matrix
    if use_raw is None and adata.raw is not None:
        genes_not_in_var = [name for name in var_names_list if name not in adata.var_names]
        if any(name in adata.raw.var_names for name in genes_not_in_var):
            use_raw = True

    if use_raw and adata.raw is not None:
        matrix = adata.raw.X
        var_names_idx = [adata.raw.var_names.get_loc(name) for name in var_names_list]
    else:
        matrix = adata.X if layer is None else adata.layers[layer]
        var_names_idx = [adata.var_names.get_loc(name) for name in var_names_list]

    # 4. Means and fractions
    means = np.zeros((len(cats), len(var_names_list)))
    fractions = np.zeros_like(means)

    for i, group in enumerate(cats):
        mask = (adata.obs[groupby] == group).values
        group_matrix = matrix[mask][:, var_names_idx]

        if group_matrix.shape[0] > 0:
            if mean_only_expressed:
                expressed = group_matrix > expression_cutoff
                means[i] = np.array(
                    [
                        group_matrix[:, j][expressed[:, j]].mean() if expressed[:, j].any() else 0
                        for j in range(group_matrix.shape[1])
                    ]
                )
            else:
                means[i] = np.asarray(np.mean(group_matrix, axis=0)).ravel()
            fractions[i] = np.asarray(np.mean(group_matrix > expression_cutoff, axis=0)).ravel()

    if standard_scale == "group":
        row_range = means.max(axis=1, keepdims=True) - means.min(axis=1, keepdims=True)
        row_range[row_range == 0] = 1
        means = (means - means.min(axis=1, keepdims=True)) / row_range
    elif standard_scale == "var":
        col_range = means.max(axis=0) - means.min(axis=0)
        col_range[col_range == 0] = 1
        means = (means - means.min(axis=0)) / col_range

    # 5. Label colors
    cat_colors_dict = _get_label_colors(adata, cats, obs_key=groupby)
    var_colors_dict = _get_label_colors(adata, var_names_list, obs_key=None)
    if y_label_colors is not None:
        cat_colors_dict.update(y_label_colors)
    if x_label_colors is not None:
        var_colors_dict.update(x_label_colors)

    # 6. Swap axes
    if swap_axes:
        matrix_means = means.T
        matrix_fractions = fractions.T
        row_items, col_items = var_names_list, cats
        row_groups, col_groups = var_groups, obs_groups
        row_colors_dict, col_colors_dict = var_colors_dict, cat_colors_dict
        row_order = list(var_names.keys()) if isinstance(var_names, dict) else None
        col_order = list(categories_order.keys()) if isinstance(categories_order, dict) else None
    else:
        matrix_means = means
        matrix_fractions = fractions
        row_items, col_items = cats, var_names_list
        row_groups, col_groups = obs_groups, var_groups
        row_colors_dict, col_colors_dict = cat_colors_dict, var_colors_dict
        row_order = list(categories_order.keys()) if isinstance(categories_order, dict) else None
        col_order = list(var_names.keys()) if isinstance(var_names, dict) else None

    h, w = matrix_means.shape
    height = h / 3 if figsize is None else figsize[1] * 0.7 * (h / max(h, w))
    width = w / 3 if figsize is None else figsize[0] * 0.7 * (w / max(h, w))

    m = ma.SizedHeatmap(
        size=matrix_fractions,
        color=matrix_means,
        height=height,
        width=width,
        cmap=cmap,
        edgecolor="black",
        size_legend_kws=dict(
            colors="#c2c2c2",
            title=size_title,
            labels=[f"{int(x * 100)}%" for x in [0.2, 0.4, 0.6, 0.8, 1.0]],
            show_at=[0.2, 0.4, 0.6, 0.8, 1.0],
            fontsize=fontsize,
            ncol=3,
        ),
        color_legend_kws=dict(
            title=colorbar_title,
            fontsize=fontsize,
            orientation="horizontal",
        ),
    )

    if col_groups:
        m.group_cols(col_groups, order=col_order)
    if row_groups:
        m.group_rows(row_groups, order=row_order)

    m.add_bottom(mp.Colors(col_items, palette=col_colors_dict), pad=0.05, size=0.15)
    m.add_bottom(mp.Labels(col_items, rotation=90, align="top", fontsize=fontsize), pad=0.1)
    m.add_left(
        CircleLabels(
            labels=row_items,
            split_groups=row_groups,
            palette=row_colors_dict,
            dot_size=100,
            fontsize=fontsize,
            text_color="black",
        ),
        pad=0.1,
        size=0.3,
    )
    m.add_legends(box_padding=2)

    fig = _marsilea_figure(m)

    if return_fig:
        return fig
    if show is not True:
        return m
    return None
