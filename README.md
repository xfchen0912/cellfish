# cellfish

Personal single-cell helpers for scRNA-seq, scATAC-seq / multiome, and (later) spatial.

```python
import cellfish as cf

cf.pl.setup_style()
cf.data.require_obs(adata, ["cell_type"])
cf.io.write_h5_safe(adata, path)
cf.ext.drvi.plot_latent_heatmap(...)  # after migration
```

Install (editable):

```bash
pip install -e ".[dev]"
```

Layout and migration notes: [PLAN.md](PLAN.md).

## What this package does

- Sanitize and write AnnData / MuData
- Check `obs` / `obsm` / `var` / layers
- Join paired modalities (RNA ↔ ATAC, later RNA ↔ spatial)
- Publication style (fonts, PDF-safe save)
- Thin wrappers around analysis tools under `ext/<tool>/`

## What it does not do

- QC, integration, peak calling, or spaceranger pipelines
- Model training (leave that in notebooks / shell)
- Project paths, paper palettes, or figure-specific panels

Those stay in the analysis repository.

## Layout

```text
src/cellfish/
├── io/          # write hygiene
├── data/        # checks, markers, pairing
├── plot/        # style, palettes, generic plots (as sch.pl)
├── stats/       # thin stats glue
└── ext/         # one folder per algorithm (lazy import)
```

Layer 1 (`io` / `data` / `plot` / `stats`) must not import `ext`.
Algorithm modules may call layer 1.

## Adding an algorithm

1. Create `src/cellfish/ext/<tool>/__init__.py` plus one implementation file
2. Keep compute and plotting together; reuse `cf.pl` for fonts / palettes / grids
3. Put optional dependencies in `pyproject.toml` extras and import them inside functions
4. Add a smoke test with a tiny simulated object
5. Do not add `plot/_<tool>.py` and do not change layer 1 unless the logic is shared by two tools
