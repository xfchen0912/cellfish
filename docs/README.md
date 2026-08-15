# cellfish

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/xfchen0912/cellfish/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/cellfish

Personal single-cell helpers for scRNA-seq, scATAC-seq / multiome, and (later) spatial.

```python
import cellfish as cf

cf.pl.setup_style()
cf.data.require_obs(adata, ["cell_type"])
cf.pl.embedding(adata, basis="X_umap", color="cell_type", palette=MY_PALETTE)
cf.pl.embedding(adata, basis="spatial", color="cell_type", palette=MY_PALETTE)
cf.io.write_h5_safe(adata, path)
```

## What this package does

- Sanitize and write AnnData / MuData
- Check `obs` / `obsm` / `var` / layers
- Join paired modalities (RNA ↔ ATAC, later RNA ↔ spatial)
- Publication plotting (`cf.pl.embedding`, palettes, plot1cell, proportions, …)
- Thin wrappers around analysis tools under `ext/<tool>/`

## What it does not do

- QC, integration, peak calling, or spaceranger pipelines
- Model training (leave that in notebooks / shell)
- Project paths, paper palettes, or figure-specific panels

Those stay in the analysis repository.

## Installation

You need Python 3.11 or newer.

1. Editable install from a clone:

```bash
git clone https://github.com/xfchen0912/cellfish.git
cd cellfish
pip install -e ".[dev]"
```

2. Optional extras:

```bash
pip install -e ".[plot]"   # marsilea dotplots, rich, fonttools
pip install -e ".[doc]"    # Sphinx documentation
```

3. Latest development version:

```bash
pip install "git+https://github.com/xfchen0912/cellfish.git@main"
```

## Release notes

See the [changelog][].

## Contact

If you found a bug, please use the [issue tracker][].

[issue tracker]: https://github.com/xfchen0912/cellfish/issues
[tests]: https://github.com/xfchen0912/cellfish/actions
[documentation]: https://cellfish.readthedocs.io
[changelog]: https://cellfish.readthedocs.io/en/latest/changelog.html
