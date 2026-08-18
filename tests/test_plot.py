import matplotlib

matplotlib.use("Agg")

import warnings

import numpy as np
from anndata import AnnData

from cellfish.plot import (
    add_contour,
    embedding,
    get_cluster_proportions,
    plot_cluster_proportions,
    reorder_and_set_palettes,
    umap,
)


def _tiny() -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(rng.normal(size=(40, 5)))
    adata.obs_names = [f"c{i}" for i in range(40)]
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.obs["cell_type"] = (["A"] * 20) + (["B"] * 20)
    adata.obs["sample"] = (["s1"] * 10) + (["s2"] * 10) + (["s1"] * 10) + (["s2"] * 10)
    adata.obs["qc"] = rng.normal(loc=5, scale=1, size=40)
    adata.obsm["X_umap"] = rng.normal(size=(40, 2))
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(40, 2))
    return adata


def test_embedding_basis_alias():
    adata = _tiny()
    ax = embedding(adata, basis="umap", color="cell_type", show=False)
    assert ax is not None
    ax2 = embedding(adata, basis="spatial", color="cell_type", show=False)
    assert ax2 is not None
    umap(adata, color="cell_type", show=False)
    from cellfish.plot._embedding import embedding as public_embedding
    from cellfish.plot._single import embedding as style_embedding

    assert public_embedding is style_embedding


def test_embedding_axis_and_colorbar_types():
    adata = _tiny()
    ax = embedding(adata, basis="X_umap", color="cell_type", axis_type="arrow", show=False)
    assert ax is not None
    assert not ax.get_xaxis().get_visible()
    ax_boxed = embedding(adata, basis="X_umap", color="cell_type", axis_type="boxed", show=False)
    assert ax_boxed is not None
    ax_hidden = embedding(adata, basis="X_umap", color="cell_type", axis_type="hidden", show=False)
    assert ax_hidden is not None
    assert not ax_hidden.get_xaxis().get_visible()
    ax_gene = embedding(adata, basis="X_umap", color="qc", colorbar_type="standard", show=False)
    assert ax_gene is not None


def test_frameon_deprecated():
    adata = _tiny()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embedding(adata, basis="X_umap", color="cell_type", frameon=False, show=False)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("frameon" in str(w.message) for w in caught)


def test_add_contour():
    adata = _tiny()
    ax = embedding(adata, basis="X_umap", show=False)
    add_contour(ax, adata, groupby="cell_type", clusters=["A"], basis="X_umap")


def test_proportions_and_palette_dict():
    adata = _tiny()
    reorder_and_set_palettes(adata, "cell_type", palette={"A": "#ff0000", "B": "#00ff00"})
    assert list(adata.uns["cell_type_colors"]) == ["#ff0000", "#00ff00"]
    props = get_cluster_proportions(adata, cluster_key="cell_type", sample_key="sample")
    assert props.shape[0] == 2
    fig = plot_cluster_proportions(props, cluster_palette={"A": "#ff0000", "B": "#00ff00"}, show=False)
    assert fig is not None
