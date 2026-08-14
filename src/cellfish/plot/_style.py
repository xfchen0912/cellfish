"""Figure defaults and PDF-safe saving. No global settings singleton."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ._fonts import validate_and_load_fonts


def setup_style(
    *,
    font: str = "Arial",
    dpi: int = 100,
    font_dir: str = "~/tmp/fonts/",
    scanpy_defaults: bool = True,
) -> list[str]:
    """Load fonts if present and set publication rcParams (PDF fonttype 42)."""
    loaded = validate_and_load_fonts([font], font_dir=font_dir)
    family = font if (loaded or _font_available(font)) else "sans-serif"
    plt.rcParams.update(
        {
            "font.family": family,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": dpi,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    if scanpy_defaults:
        try:
            import scanpy as sc

            sc.settings.verbosity = 1
            sc.settings.set_figure_params(dpi=dpi, facecolor="white", frameon=False)
        except Exception:
            pass
    return loaded


def savefig(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    also_png: bool = True,
) -> Path:
    """Save ``fig``; if path is PDF/SVG, also write a PNG sidecar unless disabled."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    if also_png and out.suffix.lower() != ".png":
        fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    return out


def _font_available(name: str) -> bool:
    from matplotlib.font_manager import fontManager

    return any(name.lower() in f.name.lower() for f in fontManager.ttflist)
