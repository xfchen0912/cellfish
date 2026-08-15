"""Render gallery thumbnails used by docs/gallery.md.

Uses the full epithelial snRNA AnnData (all cells). Only marker genes are
loaded into memory; embeddings and obs come with the slice.

Run from the cellfish repo root::

    python docs/scripts/make_gallery.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from PIL import Image

import cellfish as cf

warnings.filterwarnings("ignore")
plt.show = lambda *args, **kwargs: None  # noqa: ARG005

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_static" / "img" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = Path(
    os.environ.get(
        "CELLFISH_GALLERY_H5AD",
        "/mnt/TrueNas/project/chenxufeng/Data/NAFLD_HCC/1_AnnData/rna/rna_sub/Epithelial_sn_glue_fil.h5ad",
    )
)

# Analysis-repo palettes (passed in; not shipped in cellfish).
CLUSTER_PALETTE = {
    "Hep_PV": "#EDBAB9",
    "Hep_injury1": "#ECA121",
    "Hep_injury2": "#DF6B3A",
    "Tumor": "#DE5E6D",
    "Tumor_prolife": "#7B0B31",
    "Hep_MID": "#68AA7E",
    "Hep_CV": "#8497C2",
    "Hep_periCV": "#01ACDD",
    "Hep_mt": "#60a7a6",
}
CLUSTER_ORDER = (
    "Hep_periCV",
    "Hep_CV",
    "Hep_MID",
    "Hep_PV",
    "Hep_injury1",
    "Hep_injury2",
    "Tumor",
    "Tumor_prolife",
    "Hep_mt",
)
DISEASE_PALETTE = {
    "Healthy": "#0067AC",
    "NAFLD": "#ECA121",
    "NAFLD-HCC": "#E0622D",
}
TIME_ORDER = ("0w", "10w", "16w", "20w", "24w", "28w", "40w")
MARKERS = ["Alb", "Cyp2e1", "Glul", "Pck1", "Hal", "Sds", "Gpc3", "Krt7", "Epcam", "Asgr1"]
GROUPBY = "cell_type_highres"

# Gallery cards are 4:3 landscape.
ASPECT = 4 / 3
FIGSIZE = (5.6, 4.2)
SQUARE = (4.2, 4.2)


def _prepare(src: sc.AnnData) -> sc.AnnData:
    genes = [g for g in MARKERS if g in src.var_names]
    if not genes:
        genes = list(src.var_names[:8])
    adata = src[:, genes].to_memory()
    for key in ("X_umap", "X_umap2d", "X_pca"):
        if key in src.obsm:
            adata.obsm[key] = np.asarray(src.obsm[key])

    order = [c for c in CLUSTER_ORDER if c in set(adata.obs[GROUPBY].astype(str))]
    adata.obs[GROUPBY] = adata.obs[GROUPBY].astype(str).astype("category").cat.set_categories(order)
    if "Time" in adata.obs:
        present = [t for t in TIME_ORDER if t in set(adata.obs["Time"].astype(str))]
        adata.obs["Time"] = adata.obs["Time"].astype(str).astype("category").cat.set_categories(present)
    if "Disease_Status" in adata.obs:
        d_order = [d for d in DISEASE_PALETTE if d in set(adata.obs["Disease_Status"].astype(str))]
        adata.obs["Disease_Status"] = (
            adata.obs["Disease_Status"].astype(str).astype("category").cat.set_categories(d_order)
        )

    cf.pl.reorder_and_set_palettes(adata, GROUPBY, palette=CLUSTER_PALETTE)
    if "Disease_Status" in adata.obs:
        cf.pl.reorder_and_set_palettes(adata, "Disease_Status", palette=DISEASE_PALETTE)

    x_max = float(np.asarray(adata.X.max() if hasattr(adata.X, "max") else np.max(adata.X)))
    if x_max > 30:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print(f"full epithelial: {adata.n_obs} cells × {adata.n_vars} genes (X max={x_max:.2f})")
    return adata


def _pad_to_aspect(path: Path, aspect: float = ASPECT) -> None:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w / h > aspect:
        tw, th = w, int(round(w / aspect))
    else:
        th, tw = h, int(round(h * aspect))
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    canvas.paste(im, ((tw - w) // 2, (th - h) // 2))
    canvas.save(path)


def _save(name: str, fig=None) -> None:
    if fig is None:
        fig = plt.gcf()
    if isinstance(fig, (list, tuple, np.ndarray)):
        fig = np.ravel(fig)[0]
    if hasattr(fig, "figure") and not hasattr(fig, "savefig"):
        fig = fig.figure
    out = OUT / f"{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close("all")
    _pad_to_aspect(out)
    print("ok", name)


def _run(name, fn):
    try:
        fn()
    except Exception as exc:
        plt.close("all")
        print(f"skip {name}: {type(exc).__name__}: {exc}")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Gallery source h5ad not found: {SOURCE}")
    cf.pl.setup_style(scanpy_defaults=False, dpi=110)
    adata = _prepare(sc.read_h5ad(SOURCE, backed="r"))
    markers = [g for g in MARKERS if g in adata.var_names]
    zonation = [c for c in ("Hep_periCV", "Hep_PV") if c in set(adata.obs[GROUPBY].astype(str))]

    def embedding():
        ax = cf.pl.embedding(adata, basis="X_umap", color=GROUPBY, show=False, frameon="small")
        _save("embedding", ax)

    def numbered():
        ax = cf.pl.embedding_numbered(
            adata, basis="X_umap", color=GROUPBY, legend_title="Cell type", show=False, frameon="small"
        )
        _save("embedding_numbered", ax)

    def disease():
        ax = cf.pl.embedding(adata, basis="X_umap", color="Disease_Status", show=False, frameon="small")
        _save("spatial", ax)

    def contour():
        ax = cf.pl.embedding(adata, basis="X_umap", color=GROUPBY, show=False, frameon="small")
        cf.pl.add_contour(ax, adata, groupby=GROUPBY, clusters=zonation, basis="X_umap")
        _save("contour", ax)

    def proportions():
        props = cf.pl.get_cluster_proportions(adata, cluster_key=GROUPBY, sample_key="Sample")
        fig = cf.pl.plot_cluster_proportions(
            props, cluster_palette=CLUSTER_PALETTE, figsize=FIGSIZE, show=False
        )
        _save("proportions", fig)

    def alluvial():
        cf.pl.cell_alluvial(
            adata,
            GROUPBY,
            "Time",
            mode="proportion",
            groupby_order=list(TIME_ORDER),
            figsize=FIGSIZE,
            legend=True,
        )
        _save("alluvial")

    def stackarea():
        cf.pl.cellstackarea(adata, GROUPBY, "Time", groupby_li=list(TIME_ORDER), figsize=FIGSIZE, legend=True)
        _save("stackarea")

    def violin():
        cf.pl.violin_box(adata, keys=markers[0], groupby=GROUPBY, show=False, figsize=FIGSIZE)
        _save("violin_box")

    def bardot():
        cf.pl.bardotplot(adata, groupby=GROUPBY, color=markers[0], figsize=FIGSIZE, xticks_rotation=45)
        _save("bardotplot")

    def ridge():
        g = cf.pl.ridgeplot(adata, metric="pct_counts_mt", groupby=GROUPBY, palette=CLUSTER_PALETTE, show=False)
        _save("ridgeplot", g.figure)

    def p1c():
        cf.pl.plot1cell(
            adata,
            clusters=GROUPBY,
            tracks=["Disease_Status", "Time"],
            point_size=1.2,
            point_alpha=0.25,
            show=False,
            figsize=SQUARE,
        )
        _save("plot1cell")

    def palettes():
        shown = {k: CLUSTER_PALETTE[k] for k in CLUSTER_ORDER if k in set(adata.obs[GROUPBY].astype(str))}
        cf.pl.show_palette(shown, title="Epithelial clusters")
        _save("palettes")

    def dots():
        fig = cf.pl.dotplot(adata, var_names=markers[:6], groupby=GROUPBY)
        _save("dotplot", fig if hasattr(fig, "savefig") else None)

    for name, fn in [
        ("embedding", embedding),
        ("embedding_numbered", numbered),
        ("spatial", disease),
        ("contour", contour),
        ("proportions", proportions),
        ("alluvial", alluvial),
        ("stackarea", stackarea),
        ("violin_box", violin),
        ("bardotplot", bardot),
        ("ridgeplot", ridge),
        ("plot1cell", p1c),
        ("palettes", palettes),
        ("dotplot", dots),
    ]:
        _run(name, fn)

    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
