"""Ridge plots of an ``obs`` metric."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from scipy.stats import median_abs_deviation

from cellfish.data import require_obs

from ._style import savefig


def ridgeplot(
    adata: AnnData,
    metric: str,
    groupby: str = "Sample",
    mad: bool = False,
    nmads: int = 5,
    vmin: Union[str, float, None] = None,
    vmax: Union[str, float, None] = None,
    palette: str | dict | None = "tab20",
    show: bool = True,
    save: str | Path | None = None,
):
    """Ridge / overlapping KDE of ``adata.obs[metric]`` grouped by ``groupby``."""
    require_obs(adata, [metric, groupby])
    df = adata.obs[[metric, groupby]]

    if isinstance(vmin, str) and vmin.startswith("p"):
        vmin = float(np.percentile(df[metric], float(vmin[1:])))
    if isinstance(vmax, str) and vmax.startswith("p"):
        vmax = float(np.percentile(df[metric], float(vmax[1:])))

    x_min = 0.0 if vmin is None else float(vmin)
    x_max = float(df[metric].max() if vmax is None else vmax)

    g = sns.FacetGrid(df, row=groupby, hue=groupby, aspect=15, height=0.5, palette=palette)
    g.map(sns.kdeplot, metric, clip_on=False, fill=True, alpha=1, linewidth=1.5, clip=(x_min, x_max))
    g.map(sns.kdeplot, metric, clip_on=False, color="w", lw=2, clip=(x_min, x_max))
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def _label(_x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(_label, metric)
    g.figure.subplots_adjust(hspace=-0.6)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    for ax in g.axes.flat:
        ax.set_xlim(x_min, x_max)

    if mad:
        med = np.median(df[metric])
        spread = median_abs_deviation(df[metric])
        lo, hi = med - nmads * spread, med + nmads * spread
        for ax in g.axes.flat:
            ax.axvline(x=lo, color="r", linestyle="-")
            ax.axvline(x=hi, color="r", linestyle="-")

    if save is not None:
        savefig(g.figure, save)
    if show:
        plt.show()
    return g
