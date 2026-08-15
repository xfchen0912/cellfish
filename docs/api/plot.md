# Plotting

Publication plots under `cf.pl`. Tissue coordinates use `embedding(..., basis="spatial")`;
there is no separate spatial plotting module.

Marsilea-based `dotplot` / `grid_dotplot` need `pip install cellfish[plot]`.

## Style and fonts

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.setup_style
    plot.savefig
    plot.font_signature
    plot.validate_and_load_fonts
    plot.export_mplstyle
```

## Palettes

Project-specific colors stay in the analysis repo. Pass them with `palette=` or `group_color_dict=`.

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.get_palette
    plot.create_palette_from_types
    plot.order_labels
    plot.reorder_and_set_palettes
    plot.show_color
    plot.show_palette
    plot.list_available_palettes
```

## Embedding

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.embedding
    plot.embedding_numbered
    plot.umap
    plot.tsne
    plot.pca
    plot.mde
    plot.embedding_celltype
    plot.embedding_adjust
    plot.embedding_density
    plot.ConvexHull
    plot.add_arrow
```

## Composition

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.get_cluster_proportions
    plot.plot_cluster_proportions
    plot.cellproportion
    plot.cellstackarea
    plot.cell_alluvial
```

## Categorical / statistical plots

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.bardotplot
    plot.single_group_boxplot
    plot.plot_boxplots
    plot.violin_box
    plot.violin_old
    plot.dotplot_doublegroup
    plot.ridgeplot
```

## Contours and obs scatter

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.add_contour
    plot.contour
    plot.plot_scatter
    plot.obs_scatter
```

## plot1cell

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.plot1cell
    plot.plot1cell_atlas_meta_rings
    plot.simulate_atlas_anndata
```

## Dotplots (marsilea)

These attributes are loaded lazily. Install `cellfish[plot]` first.

```{eval-rst}
.. currentmodule:: cellfish

.. autosummary::
    :toctree: generated

    plot.dotplot
    plot.grid_dotplot
    plot.CircleLabels
    plot.rank_genes_groups_dotplot
```
