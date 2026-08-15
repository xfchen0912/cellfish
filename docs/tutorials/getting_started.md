# Getting started

This page uses a tiny simulated {class}`~anndata.AnnData`. Swap it for your own object.

## Install

```bash
pip install -e ".[dev]"
```

## Load and check

```python
import numpy as np
from anndata import AnnData
import cellfish as cf

rng = np.random.default_rng(0)
adata = AnnData(rng.normal(size=(80, 20)))
adata.obs_names = [f"c{i}" for i in range(80)]
adata.var_names = [f"g{i}" for i in range(20)]
adata.obs["cell_type"] = (["Hepatocyte"] * 40) + (["Macrophage"] * 40)
adata.obs["sample"] = (["s1"] * 20 + ["s2"] * 20) * 2
adata.obsm["X_umap"] = rng.normal(size=(80, 2))
adata.obsm["spatial"] = rng.uniform(0, 10, size=(80, 2))

cf.data.require_obs(adata, ["cell_type", "sample"])
cf.data.require_obsm(adata, "umap")  # also accepts X_umap
```

## Style and palettes

Paper palettes stay in the analysis repo. Pass a dict:

```python
MY_PALETTE = {"Hepatocyte": "#1F577B", "Macrophage": "#E069A6"}

cf.pl.setup_style()
cf.pl.reorder_and_set_palettes(adata, "cell_type", palette=MY_PALETTE)
```

## Embedding (UMAP or tissue)

The same function draws both. Change `basis=`.

```python
ax = cf.pl.embedding(adata, basis="X_umap", color="cell_type", show=False)
cf.pl.add_contour(ax, adata, groupby="cell_type", clusters=["Hepatocyte"], basis="X_umap")

cf.pl.embedding(adata, basis="spatial", color="cell_type", show=False)
```

`cf.pl.umap(...)` is a shortcut that requires `adata.obsm["X_umap"]`.

## Composition

```python
props = cf.pl.get_cluster_proportions(
    adata, cluster_key="cell_type", sample_key="sample"
)
fig = cf.pl.plot_cluster_proportions(
    props, cluster_palette=MY_PALETTE, show=False
)
cf.pl.savefig(fig, "figures/cell_type_proportions.pdf")
```

## Write AnnData

```python
cf.io.write_h5_safe(adata, "adata.clean.h5ad")
```

## Next

- API: {doc}`/api/index`
- Adding an algorithm under `cf.ext`: {doc}`/contributing`
