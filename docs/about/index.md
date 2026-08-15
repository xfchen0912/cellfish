# About cellfish

`cellfish` is a personal, reusable layer on top of {cite:p}`wolf2018scanpy` and {cite:p}`virshup2023anndata`.
It is not a pipeline and not a paper-specific figure pack.

Import it as:

```python
import cellfish as cf
```

The public surface is five namespaces: `cf.io`, `cf.data`, `cf.pl`, `cf.stats`, `cf.ext`.

```{mermaid}
flowchart TD
    subgraph layer1 [Layer 1 — always reusable]
        io["cf.io"]
        data["cf.data"]
        pl["cf.pl"]
        stats["cf.stats"]
    end
    subgraph ext [Layer 2 — one folder per algorithm]
        tools["cf.ext.milo / drvi / scenicplus / …"]
    end
    subgraph project [Layer 3 — stays in the analysis repo]
        palettes["paper palettes, DATA_DIR, Fig panels"]
    end
    tools --> layer1
    project --> layer1
    project --> tools
```

**Dependency direction:** `ext` may call layer 1. Layer 1 must not import `ext`.

## Three layers

| Layer | What belongs here | After switching projects |
| --- | --- | --- |
| 1 — `io` / `data` / `plot` / `stats` | Algorithm-agnostic write hygiene, checks, pairing, fonts, generic plots | Still used |
| 2 — `ext/<tool>/` | One algorithm per folder; compute and that algorithm’s plots together | Skip the extra if you do not install the tool |
| 3 — analysis repo | `DATA_DIR`, paper palettes, cohort registries, figure functions | Must change |

A helper belongs in layer 1 if you would still use it on a new species, without DRVI / ChromBPNet, and without the MASLD-HCC color dictionary.

## Coordinates, not extra packages

UMAP, tissue, and (later) genomic loci are different `basis=` / coordinate sources, not three top-level packages.

| Space | Typical key | Background |
| --- | --- | --- |
| Cell embedding | `adata.obsm["X_umap"]` | none |
| Tissue | `adata.obsm["spatial"]` | optional H&E / IF |
| Genomic locus | chrom, start, end | optional bigWig / contrib |

`var` is not assumed to be genes (peaks, cCREs, motifs, bins are fine).
Objects today are AnnData and MuData. SpatialData is not a requirement until spatial work lands.

## Project palettes

Paper colors such as disease or cluster dictionaries stay in the analysis repository.
Pass them in:

```python
cf.pl.embedding(adata, basis="X_umap", color="cell_type", palette=MY_PALETTE)
cf.pl.reorder_and_set_palettes(adata, "cell_type", palette=MY_PALETTE)
```
