"""
Atlas-style plot1cell figure with original plot1cell-like metadata rings.

This module provides:
1. plot1cell_atlas_meta_rings:
   Main function for an AnnData object. It draws:
   - central UMAP/t-SNE scatter
   - KDE contour
   - cell labels
   - multi-layer metadata rings, plot1cell-style
   - outer major-class arcs
   - optional inset UMAPs
   - non-overlapping legends

2. simulate_atlas_anndata:
   A small helper to generate an AnnData-like simulated dataset for testing.

Expected AnnData fields:
    adata.obsm[basis]                  # 2D coordinates, e.g. X_umap
    adata.obs[clusters]                # cell-type / fine cluster
    adata.obs[supergroup]              # major class / coarse group
    adata.obs[meta_rings columns]      # metadata shown as circular rings
    adata.obs[inset_cols columns]      # metadata shown as inset UMAPs

Example:
    fig, ax, df = plot1cell_atlas_meta_rings(
        adata,
        clusters="cell_type",
        supergroup="major_class",
        basis="X_umap",
        meta_rings=("cell_type", "species", "assay", "dataset"),
        inset_cols=("species", "assay", "cell_type", "dataset"),
        font_family="Arial",
    )
    fig.savefig("atlas_plot.pdf", bbox_inches="tight")
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from matplotlib import font_manager
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------
# Small AnnData-like container for simulation/testing only
# ---------------------------------------------------------------------
class MockAnnData:
    """Minimal AnnData-like container used by simulate_atlas_anndata."""

    def __init__(self, obs: pd.DataFrame, obsm: Optional[dict] = None, uns: Optional[dict] = None):
        self.obs = obs
        self.obsm = obsm or {}
        self.uns = uns or {}
        self.obs_names = obs.index.astype(str)


# ---------------------------------------------------------------------
# Font and plotting helpers
# ---------------------------------------------------------------------
def set_preferred_font(font_family: Union[str, Sequence[str]] = "Arial") -> str:
    """
    Set a preferred sans-serif font.

    In many Linux environments Arial is not installed. This function tries:
    requested font -> Arimo -> Liberation Sans -> DejaVu Sans.

    Returns the actual first available font name.
    """
    if isinstance(font_family, str):
        requested = [font_family]
    else:
        requested = list(font_family)

    fallback = ["Arimo", "Liberation Sans", "DejaVu Sans"]
    candidates = requested + [x for x in fallback if x not in requested]
    available = {f.name for f in font_manager.fontManager.ttflist}

    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:
        chosen = "DejaVu Sans"

    plt.rcParams["font.family"] = chosen
    # Keep text editable in vector files.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    return chosen


def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str)


def _levels(series: pd.Series, explicit_order: Optional[Sequence] = None) -> list[str]:
    if explicit_order is not None:
        return [str(x) for x in explicit_order]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories]
    return list(pd.Series(series.astype(str)).drop_duplicates())


def _transform_coordinates(v: np.ndarray, zoom: float) -> np.ndarray:
    """Same idea as plot1cell: center one axis and scale independently."""
    lo, hi = float(np.min(v)), float(np.max(v))
    centred = v - 0.5 * (lo + hi)
    m = float(np.max(np.abs(centred))) or 1.0
    return centred * zoom / m


def _resolve_palette(n: int, palette=None) -> list:
    if palette is None:
        cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar", n)
        return [cmap(i) for i in range(n)]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette, n)
        return [cmap(i) for i in range(n)]
    palette = list(palette)
    if len(palette) == 0:
        raise ValueError("palette cannot be empty")
    return [palette[i % len(palette)] for i in range(n)]


def _palette_for(
    levels: Sequence[str],
    palette=None,
    adata=None,
    key: Optional[str] = None,
    fallback: str = "tab20",
) -> dict[str, object]:
    levels = [str(x) for x in levels]

    if isinstance(palette, Mapping):
        return {str(k): v for k, v in palette.items()}

    if palette is not None:
        colors = _resolve_palette(len(levels), palette)
        return dict(zip(levels, colors))

    if adata is not None and key is not None and hasattr(adata, "uns"):
        uns_key = f"{key}_colors"
        if uns_key in adata.uns:
            stored = list(adata.uns[uns_key])
            if len(stored) >= len(levels):
                return dict(zip(levels, stored[: len(levels)]))

    colors = _resolve_palette(len(levels), fallback)
    return dict(zip(levels, colors))


def _run_length(vals: np.ndarray):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return np.array([]), np.array([]), np.array([])
    change = np.concatenate([[True], vals[1:] != vals[:-1]])
    starts = np.flatnonzero(change)
    lengths = np.diff(np.concatenate([starts, [len(vals)]]))
    values = vals[starts]
    return starts, lengths, values


def _draw_wedge(
    ax: Axes,
    r: float,
    theta1: float,
    theta2: float,
    width: float,
    facecolor,
    *,
    edgecolor: str = "white",
    linewidth: float = 0.6,
    zorder: float = 4,
):
    ax.add_patch(
        Wedge(
            (0, 0),
            r,
            theta1,
            theta2,
            width=width,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            joinstyle="miter",
            zorder=zorder,
        )
    )


def _add_arc_label(
    ax: Axes,
    text: str,
    angle_deg: float,
    radius: float,
    *,
    fontsize: float = 15,
    fontweight: str = "normal",
    zorder: float = 10,
):
    rad = np.deg2rad(angle_deg)
    x = radius * np.cos(rad)
    y = radius * np.sin(rad)
    rot = angle_deg - 90
    if 90 < (angle_deg % 360) < 270:
        rot += 180
    ax.text(
        x,
        y,
        str(text),
        ha="center",
        va="center",
        rotation=rot,
        rotation_mode="anchor",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=zorder,
    )


def _auto_major_arc_ranges(groups: Sequence[str]) -> dict[str, Tuple[float, float]]:
    """
    Reasonable default positions for common atlas groups.
    For unknown groups, distribute arcs around the circle.
    """
    groups = [str(g) for g in groups]

    canonical = {
        "Embryonic": (108, 145),
        "Neuronal": (48, 101),
        "Epithelium": (150, 206),
        "Mesenchyme": (-38, 27),
        "Blood": (220, 310),
    }
    if all(g in canonical for g in groups):
        return {g: canonical[g] for g in groups}

    # Generic fallback: distribute group arcs evenly with gaps.
    gap = 10.0
    usable = 360.0 - gap * len(groups)
    width = usable / max(len(groups), 1)
    out = {}
    cur = 90.0
    for g in groups:
        out[g] = (cur, cur + width)
        cur += width + gap
    return out


def _add_umap_inset(
    fig: Figure,
    df: pd.DataFrame,
    col: str,
    rect: Sequence[float],
    title: str,
    *,
    levels: Optional[Sequence[str]] = None,
    palette=None,
    point_size: float = 0.13,
    alpha: float = 0.80,
    show_legend: bool = False,
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_loc: str = "center left",
    legend_ncol: int = 1,
):
    ax = fig.add_axes(rect)
    if levels is None:
        levels = _levels(df[col])
    levels = [str(x) for x in levels]
    colors = _palette_for(levels, palette, key=col, fallback="tab20")
    vals = df[col].astype(str).to_numpy()

    ax.scatter(
        df["x"],
        df["y"],
        s=point_size,
        c=[colors[v] for v in vals],
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=2)

    if show_legend:
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=5,
                markerfacecolor=colors[str(l)],
                markeredgecolor="none",
                label=str(l),
            )
            for l in levels
        ]
        kwargs = dict(
            handles=handles,
            frameon=True,
            fancybox=False,
            framealpha=0.94,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=7.2,
            loc=legend_loc,
            borderaxespad=0.2,
            handletextpad=0.35,
            labelspacing=0.35,
            ncol=legend_ncol,
            columnspacing=0.75,
        )
        if legend_bbox is not None:
            kwargs["bbox_to_anchor"] = legend_bbox
        leg = ax.legend(**kwargs)
        leg.set_zorder(30)
    return ax


def _make_custom_legend(
    fig: Figure,
    labels: Sequence[str],
    colors: Mapping[str, object],
    rect: Sequence[float],
    *,
    title: Optional[str] = None,
    ncol: int = 1,
    fontsize: float = 7.2,
):
    """Dedicated legend axes; useful to avoid legend-over-inset overlap."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    if title:
        ax.text(0, 1.0, title, fontsize=9, va="top", ha="left")

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=colors[str(l)],
            markeredgecolor="none",
            label=str(l),
        )
        for l in labels
    ]
    leg = ax.legend(
        handles=handles,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=fontsize,
        loc="upper left",
        bbox_to_anchor=(0, 0.86),
        borderaxespad=0,
        handletextpad=0.35,
        labelspacing=0.35,
        ncol=ncol,
        columnspacing=0.8,
    )
    leg.set_zorder(30)
    return ax


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def plot1cell_atlas_meta_rings(
    adata,
    *,
    clusters: str = "cell_type",
    supergroup: str = "major_class",
    basis: str = "X_umap",
    meta_rings: Sequence[str] = ("cell_type", "species", "assay", "dataset"),
    inset_cols: Sequence[str] = ("species", "assay", "cell_type", "dataset"),
    cluster_order: Optional[Sequence[str]] = None,
    supergroup_order: Optional[Sequence[str]] = None,
    major_arc_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    label_subset: Optional[Sequence[str]] = None,
    label_min_cells: Optional[int] = None,
    show_labels: bool = True,
    coord_scale: float = 0.82,
    point_size: float = 0.95,
    point_alpha: float = 0.58,
    kde_levels: Optional[Sequence[float]] = (0.10, 0.18, 0.29),
    kde_color: str = "#8a8a8a",
    kde_linewidth: float = 0.55,
    kde_alpha: float = 0.72,
    kde_n: int = 230,
    # ring geometry
    ring_inner_radius: float = 1.000,
    ring_widths: Optional[Mapping[str, float]] = None,
    ring_gap: float = 0.008,
    ring_edgecolor: str = "white",
    ring_linewidth: float = 0.60,
    outer_major_gap: float = 0.033,
    outer_major_width: float = 0.040,
    outer_major_label_pad: float = 0.070,
    group_gap_deg: float = 1.25,
    # palettes
    cluster_palette=None,
    supergroup_palette=None,
    ring_palettes: Optional[Mapping[str, object]] = None,
    inset_palettes: Optional[Mapping[str, object]] = None,
    # layout
    figsize: Tuple[float, float] = (14.2, 11.3),
    main_ax_rect: Sequence[float] = (0.175, 0.075, 0.675, 0.84),
    outer_xlim: Tuple[float, float] = (-1.34, 1.34),
    outer_ylim: Tuple[float, float] = (-1.26, 1.26),
    inset_layout: Optional[Mapping[str, Sequence[float]]] = None,
    show_inset_legends: Optional[Mapping[str, bool]] = None,
    dedicated_legend_layout: Optional[Mapping[str, Sequence[float]]] = None,
    panel_label: Optional[str] = "D",
    font_family: Union[str, Sequence[str]] = "Arial",
    show: bool = True,
    return_data: bool = True,
):
    """
    Draw an atlas-style plot1cell figure with original plot1cell-like inner metadata rings.

    Parameters
    ----------
    adata
        AnnData-like object with .obs and .obsm.
    clusters
        Fine-level cluster/cell-type column in adata.obs.
    supergroup
        Coarse-level group column in adata.obs.
    basis
        Coordinate key in adata.obsm.
    meta_rings
        Metadata columns drawn as circular rings. By default:
        cell_type, species, assay, dataset.
    inset_cols
        Metadata columns shown as small inset UMAPs.
    major_arc_ranges
        Optional manual arc ranges for each supergroup, e.g.
        {"Embryonic": (108,145), "Neuronal": (48,101), ...}
    ring_palettes
        Optional dict mapping ring column -> palette or {level: color}.
    inset_palettes
        Optional dict mapping inset column -> palette or {level: color}.
    font_family
        Uses Arial if installed; otherwise falls back to Arimo/Liberation Sans/DejaVu Sans.

    Returns
    -------
    fig, ax, df
        If return_data=True.
    fig, ax
        If return_data=False.
    """
    set_preferred_font(font_family)

    # Validate input
    if basis not in adata.obsm:
        raise KeyError(f"{basis!r} not found in adata.obsm")
    for col in {clusters, supergroup, *meta_rings, *inset_cols}:
        if col not in adata.obs:
            raise KeyError(f"{col!r} not found in adata.obs")

    df = adata.obs.copy()
    xy = np.asarray(adata.obsm[basis])[:, :2]
    df["x"] = _transform_coordinates(xy[:, 0], coord_scale)
    df["y"] = _transform_coordinates(xy[:, 1], coord_scale)

    cluster_levels = _levels(df[clusters], cluster_order)
    super_levels = _levels(df[supergroup], supergroup_order)

    # cluster -> supergroup mapping
    c2g_df = (
        df[[clusters, supergroup]]
        .assign(**{clusters: df[clusters].astype(str), supergroup: df[supergroup].astype(str)})
        .drop_duplicates()
    )
    n_group_per_cluster = c2g_df.groupby(clusters)[supergroup].nunique()
    bad = n_group_per_cluster[n_group_per_cluster > 1]
    if len(bad) > 0:
        raise ValueError(
            "Each cluster must map to a single supergroup. "
            f"Problematic clusters include: {', '.join(map(str, bad.index[:10]))}"
        )
    cluster_to_group = dict(zip(c2g_df[clusters], c2g_df[supergroup]))

    # Sort cluster order by supergroup, preserving user/categorical order inside group.
    group_rank = {str(g): i for i, g in enumerate(super_levels)}
    old_rank = {str(c): i for i, c in enumerate(cluster_levels)}
    cluster_levels = sorted(
        [str(c) for c in cluster_levels],
        key=lambda c: (group_rank.get(cluster_to_group.get(c), 999), old_rank.get(c, 999)),
    )

    if major_arc_ranges is None:
        major_arc_ranges = _auto_major_arc_ranges(super_levels)
    else:
        major_arc_ranges = {str(k): tuple(v) for k, v in major_arc_ranges.items()}

    missing = [g for g in super_levels if str(g) not in major_arc_ranges]
    if missing:
        raise ValueError(f"major_arc_ranges missing supergroups: {missing}")

    # Palettes
    cluster_colors = _palette_for(
        cluster_levels, cluster_palette, adata, clusters, fallback="tab20"
    )

    if supergroup_palette is None:
        canonical_major_colors = {
            "Embryonic": "#179b73",
            "Neuronal": "#6e69b8",
            "Epithelium": "#64a61f",
            "Mesenchyme": "#a87a16",
            "Blood": "#666666",
        }
        if all(str(g) in canonical_major_colors for g in super_levels):
            super_colors = {str(g): canonical_major_colors[str(g)] for g in super_levels}
        else:
            super_colors = _palette_for(super_levels, "Dark2", adata, supergroup, fallback="Dark2")
    else:
        super_colors = _palette_for(super_levels, supergroup_palette, adata, supergroup, fallback="Dark2")

    ring_palettes = dict(ring_palettes or {})
    inset_palettes = dict(inset_palettes or {})

    ring_color_maps: dict[str, dict[str, object]] = {}
    for ring in meta_rings:
        levels = _levels(df[ring])
        if ring == clusters and ring not in ring_palettes:
            ring_color_maps[ring] = cluster_colors
        else:
            ring_color_maps[ring] = _palette_for(
                levels,
                ring_palettes.get(ring, None),
                adata,
                ring,
                fallback="Set2" if ring in ("assay", "species") else "tab20",
            )

    # Allocate fine cluster arcs inside each supergroup arc.
    counts = df[clusters].astype(str).value_counts().reindex(cluster_levels).fillna(0).astype(int)
    cluster_arcs: dict[str, Tuple[float, float]] = {}
    for g in super_levels:
        g = str(g)
        cts = [c for c in cluster_levels if cluster_to_group.get(c) == g]
        if not cts:
            continue
        a0, a1 = major_arc_ranges[g]
        usable = (a1 - a0) - group_gap_deg * (len(cts) - 1)
        if usable <= 0:
            raise ValueError(f"Arc for {g!r} is too small for {len(cts)} clusters.")
        weights = np.log10(np.maximum(counts[cts].to_numpy(), 2))
        widths = weights / weights.sum() * usable
        cur = a0
        for ct, ww in zip(cts, widths):
            cluster_arcs[ct] = (cur, cur + float(ww))
            cur += float(ww) + group_gap_deg

    # Ring geometry
    if ring_widths is None:
        ring_widths = {}
    ring_bounds: dict[str, Tuple[float, float]] = {}
    r0 = ring_inner_radius
    for ring in meta_rings:
        w = float(ring_widths.get(ring, 0.032 if ring == clusters else 0.026))
        ring_bounds[ring] = (r0, r0 + w)
        r0 = r0 + w + ring_gap

    meta_outer = r0 - ring_gap
    major_r0 = meta_outer + outer_major_gap
    major_r1 = major_r0 + outer_major_width
    major_label_r = major_r1 + outer_major_label_pad

    # Figure and main axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(main_ax_rect)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(*outer_xlim)
    ax.set_ylim(*outer_ylim)

    if panel_label:
        ax.text(outer_xlim[0] + 0.06, outer_ylim[1] - 0.08, panel_label, fontsize=18, ha="left", va="center")

    # Main scatter
    vals = df[clusters].astype(str).to_numpy()
    ax.scatter(
        df["x"],
        df["y"],
        s=point_size,
        c=[cluster_colors[v] for v in vals],
        alpha=point_alpha,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # KDE contour
    if kde_levels is not None and len(df) >= 10:
        try:
            kde = gaussian_kde(np.vstack([df["x"].to_numpy(), df["y"].to_numpy()]))
            lim = coord_scale * 1.07
            gx, gy = np.mgrid[-lim:lim:complex(kde_n), -lim:lim:complex(kde_n)]
            zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            zz = zz / (zz.max() + 1e-12)
            ax.contour(
                gx,
                gy,
                zz,
                levels=sorted(kde_levels),
                colors=kde_color,
                linewidths=kde_linewidth,
                alpha=kde_alpha,
                zorder=3,
            )
        except Exception as exc:
            warnings.warn(f"KDE skipped: {exc}", stacklevel=2)

    # Dashed supergroup boundary lines
    # for g in super_levels:
    #     a0, a1 = major_arc_ranges[str(g)]
    #     for ang in (a0, a1):
    #         rr = np.deg2rad(ang)
    #         ax.plot(
    #             [0.12 * np.cos(rr), (ring_inner_radius + 0.01) * np.cos(rr)],
    #             [0.12 * np.sin(rr), (ring_inner_radius + 0.01) * np.sin(rr)],
    #             linestyle=(0, (4.2, 4.2)),
    #             color="black",
    #             lw=0.85,
    #             alpha=0.78,
    #             zorder=1,
    #         )

    # Metadata rings
    for ring in meta_rings:
        rr0, rr1 = ring_bounds[ring]
        cmap = ring_color_maps[ring]

        for ct in cluster_levels:
            a0, a1 = cluster_arcs[ct]
            sub = df[df[clusters].astype(str) == ct]
            if len(sub) == 0:
                continue

            if ring == clusters:
                _draw_wedge(
                    ax,
                    rr1,
                    a0,
                    a1,
                    rr1 - rr0,
                    cmap[ct],
                    edgecolor=ring_edgecolor,
                    linewidth=ring_linewidth + 0.10,
                    zorder=5,
                )
            else:
                # plot1cell-like behavior: sort metadata within each fine-cluster sector
                sub = sub.sort_values(ring)
                arr = sub[ring].astype(str).to_numpy()
                _, lengths, values = _run_length(arr)
                total = len(sub)
                cur = 0
                for ln, val in zip(lengths, values):
                    aa0 = a0 + (cur / total) * (a1 - a0)
                    aa1 = a0 + ((cur + ln) / total) * (a1 - a0)
                    _draw_wedge(
                        ax,
                        rr1,
                        aa0,
                        aa1,
                        rr1 - rr0,
                        cmap[str(val)],
                        edgecolor=ring_edgecolor,
                        linewidth=ring_linewidth,
                        zorder=5,
                    )
                    cur += ln

    # Subtle ring outlines per group range
    for rr0, rr1 in ring_bounds.values():
        for rr in (rr0, rr1):
            for g in super_levels:
                a0, a1 = major_arc_ranges[str(g)]
                theta = np.linspace(np.deg2rad(a0), np.deg2rad(a1), 120)
                ax.plot(rr * np.cos(theta), rr * np.sin(theta), color="#f6f6f6", lw=0.35, zorder=6)

    # Outer major arcs
    for g in super_levels:
        g = str(g)
        a0, a1 = major_arc_ranges[g]
        _draw_wedge(
            ax,
            major_r1,
            a0,
            a1,
            major_r1 - major_r0,
            super_colors[g],
            edgecolor="white",
            linewidth=1.1,
            zorder=7,
        )
        ax.add_patch(
            Wedge(
                (0, 0),
                major_r1,
                a0,
                a1,
                width=major_r1 - major_r0,
                facecolor="none",
                edgecolor="#555555",
                linewidth=0.45,
                zorder=8,
            )
        )
        _add_arc_label(ax, g, 0.5 * (a0 + a1), major_label_r, fontsize=15)

    # Labels
    if show_labels:
        if label_subset is not None:
            label_levels = [str(x) for x in label_subset]
        elif label_min_cells is not None:
            label_levels = [ct for ct in cluster_levels if counts.get(ct, 0) >= label_min_cells]
        else:
            label_levels = cluster_levels

        for ct in label_levels:
            sub = df[df[clusters].astype(str) == ct]
            if len(sub) == 0:
                continue
            ax.text(
                sub["x"].median(),
                sub["y"].median(),
                str(ct),
                ha="center",
                va="center",
                fontsize=15,
                color="black",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                zorder=9,
            )

    # Insets
    if inset_layout is None:
        inset_layout = {
            "species": (0.018, 0.700, 0.175, 0.195),
            "assay": (0.032, 0.105, 0.175, 0.195),
            "cell_type": (0.820, 0.700, 0.150, 0.195),
            "dataset": (0.820, 0.115, 0.150, 0.195),
        }
    if show_inset_legends is None:
        show_inset_legends = {col: False for col in inset_cols}

    for col in inset_cols:
        if col not in inset_layout:
            continue
        levels = _levels(df[col])
        pal = inset_palettes.get(col, ring_color_maps.get(col, None))
        _add_umap_inset(
            fig,
            df,
            col,
            inset_layout[col],
            col.replace("_", " ").title(),
            levels=levels,
            palette=pal,
            show_legend=bool(show_inset_legends.get(col, False)),
        )

    # Dedicated legends for small metadata columns, avoiding overlap.
    if dedicated_legend_layout is None:
        dedicated_legend_layout = {
            "species": (0.220, 0.710, 0.080, 0.080),
            "assay": (0.220, 0.115, 0.125, 0.140),
        }

    for col, rect in dedicated_legend_layout.items():
        if col not in df.columns:
            continue
        # Prefer ring colors; fall back to inset_palettes / adata.uns so
        # inset-only columns (e.g. Disease_Status) can still get a legend.
        if col in ring_color_maps:
            cmap = ring_color_maps[col]
        else:
            levels = _levels(df[col])
            cmap = _palette_for(
                levels,
                inset_palettes.get(col, None),
                adata,
                col,
                fallback="tab20",
            )
        _make_custom_legend(
            fig,
            _levels(df[col]),
            cmap,
            rect,
            title=None,
            ncol=1,
            fontsize=7.2,
        )

    if show:
        plt.show()

    if return_data:
        return fig, ax, df
    return fig, ax


# ---------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------
def simulate_atlas_anndata(n_cells: int = 18000, random_state: int = 11) -> MockAnnData:
    """
    Generate a simulated atlas-like AnnData object for testing the plotting function.
    """
    rng = np.random.default_rng(random_state)

    major_to_celltypes = {
        "Embryonic": ["Early embryonic cell", "Germ cell", "Epithelial progenitor"],
        "Neuronal": ["Neural progenitor", "Neuron", "Glia"],
        "Epithelium": ["Pancrea", "Kidney", "Respiratory system", "Intestine", "Liver", "Epidermal cell"],
        "Mesenchyme": [
            "Mesenchymal progenitor",
            "Bone",
            "Stromal cell",
            "Skeletal muscle",
            "Smooth muscle",
            "Pericyte",
            "Heart",
        ],
        "Blood": [
            "Blood progenitor",
            "T cell",
            "B cell",
            "Natural killer",
            "Macrophage",
            "Dendritic cell",
            "Granulocyte",
            "Megakaryocyte",
            "Erythroid",
            "Other blood cell",
            "Endothelial",
        ],
    }

    major_centers = {
        "Embryonic": (-1.2, 1.25),
        "Neuronal": (1.25, 1.35),
        "Epithelium": (-1.9, 0.0),
        "Mesenchyme": (0.65, 0.2),
        "Blood": (0.9, -1.35),
    }

    all_ct = [ct for cts in major_to_celltypes.values() for ct in cts]
    raw_sizes = rng.lognormal(mean=0.0, sigma=0.75, size=len(all_ct))
    raw_sizes = raw_sizes / raw_sizes.sum()
    sizes = np.maximum(80, np.round(raw_sizes * n_cells).astype(int))
    sizes[np.argmax(sizes)] += n_cells - sizes.sum()

    species_levels = ["human", "mouse"]
    assay_levels = ["10x", "10x multiome", "Drop-seq", "Smart-seq2", "microwell-seq", "sci-RNA-seq"]
    dataset_levels = [f"Dataset {i}" for i in range(1, 13)]

    rows = []
    xy = []
    ct_i = 0
    for major, cts in major_to_celltypes.items():
        mx, my = major_centers[major]
        local_angles = np.linspace(0, 2 * np.pi, len(cts), endpoint=False)
        local_angles += rng.normal(0, 0.15, size=len(cts))

        for j, ct in enumerate(cts):
            n = int(sizes[ct_i])
            ct_i += 1

            radius = rng.uniform(0.18, 0.65)
            cx = mx + radius * np.cos(local_angles[j]) + rng.normal(0, 0.08)
            cy = my + radius * np.sin(local_angles[j]) + rng.normal(0, 0.08)

            theta = rng.uniform(0, np.pi)
            major_axis = rng.uniform(0.10, 0.24)
            minor_axis = rng.uniform(0.025, 0.09)
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            cov = R @ np.diag([major_axis**2, minor_axis**2]) @ R.T

            pts = rng.multivariate_normal([cx, cy], cov, size=n)
            if n > 250 and rng.random() < 0.55:
                m = max(30, n // 5)
                tail_dir = np.array([np.cos(theta), np.sin(theta)])
                t = rng.uniform(0, 1.4, size=m)[:, None]
                pts[:m] = (
                    np.array([cx, cy])
                    + t * tail_dir * rng.uniform(0.18, 0.45)
                    + rng.normal(0, 0.04, size=(m, 2))
                )

            xy.append(pts)

            for _ in range(n):
                if major in ["Embryonic", "Neuronal"]:
                    species = rng.choice(species_levels, p=[0.45, 0.55])
                elif major == "Epithelium":
                    species = rng.choice(species_levels, p=[0.65, 0.35])
                else:
                    species = rng.choice(species_levels, p=[0.55, 0.45])

                if ct in ["Neuron", "Glia", "Neural progenitor", "Early embryonic cell"]:
                    assay = rng.choice(assay_levels, p=[0.20, 0.10, 0.18, 0.18, 0.10, 0.24])
                else:
                    assay = rng.choice(assay_levels, p=[0.43, 0.12, 0.16, 0.12, 0.08, 0.09])

                dataset = rng.choice(dataset_levels)
                rows.append(
                    {
                        "cell_type": ct,
                        "major_class": major,
                        "species": species,
                        "assay": assay,
                        "dataset": dataset,
                    }
                )

    xy = np.vstack(xy)
    obs = pd.DataFrame(rows, index=[f"cell_{i:05d}" for i in range(len(rows))])

    obs["cell_type"] = pd.Categorical(obs["cell_type"], categories=all_ct, ordered=True)
    obs["major_class"] = pd.Categorical(obs["major_class"], categories=list(major_to_celltypes.keys()), ordered=True)
    obs["species"] = pd.Categorical(obs["species"], categories=species_levels, ordered=True)
    obs["assay"] = pd.Categorical(obs["assay"], categories=assay_levels, ordered=True)
    obs["dataset"] = pd.Categorical(obs["dataset"], categories=dataset_levels, ordered=True)

    return MockAnnData(obs=obs, obsm={"X_umap": xy}, uns={})


if __name__ == "__main__":
    # Quick demo:
    adata = simulate_atlas_anndata(n_cells=18000, random_state=11)
    fig, ax, df = plot1cell_atlas_meta_rings(adata, show=True)
    fig.savefig("plot1cell_atlas_meta_rings_demo.png", dpi=300, bbox_inches="tight")
