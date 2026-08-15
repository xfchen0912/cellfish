::::{grid} 1 2 3 3
:gutter: 2
:class-container: sd-text-center

:::{grid-item-card} Embedding
:link: api/generated/cellfish.plot.embedding
:link-type: doc
:img-top: _static/img/gallery/embedding.png
:shadow: sm

UMAP, t-SNE, PCA, or tissue — same function, change ``basis=``.
+++
`cf.pl.embedding`
:::

:::{grid-item-card} Numbered embedding
:link: api/generated/cellfish.plot.embedding_numbered
:link-type: doc
:img-top: _static/img/gallery/embedding_numbered.png
:shadow: sm

Centroid indices with a matching right-hand legend.
+++
`cf.pl.embedding_numbered`
:::

:::{grid-item-card} Disease status
:link: api/generated/cellfish.plot.embedding
:link-type: doc
:img-top: _static/img/gallery/spatial.png
:shadow: sm

Same UMAP, colored by ``Disease_Status``.
+++
`color="Disease_Status"`
:::

:::{grid-item-card} Density contour
:link: api/generated/cellfish.plot.add_contour
:link-type: doc
:img-top: _static/img/gallery/contour.png
:shadow: sm

KDE outlines for selected clusters on any embedding.
+++
`cf.pl.add_contour`
:::

:::{grid-item-card} Proportions
:link: api/generated/cellfish.plot.plot_cluster_proportions
:link-type: doc
:img-top: _static/img/gallery/proportions.png
:shadow: sm

Stacked bars of cluster composition per sample.
+++
`cf.pl.plot_cluster_proportions`
:::

:::{grid-item-card} Alluvial
:link: api/generated/cellfish.plot.cell_alluvial
:link-type: doc
:img-top: _static/img/gallery/alluvial.png
:shadow: sm

Smooth ribbons of cell-type composition over a covariate.
+++
`cf.pl.cell_alluvial`
:::

:::{grid-item-card} Stacked area
:link: api/generated/cellfish.plot.cellstackarea
:link-type: doc
:img-top: _static/img/gallery/stackarea.png
:shadow: sm

Area chart of the same composition tables.
+++
`cf.pl.cellstackarea`
:::

:::{grid-item-card} Violin + box
:link: api/generated/cellfish.plot.violin_box
:link-type: doc
:img-top: _static/img/gallery/violin_box.png
:shadow: sm

Half-violin, box, and strip for an obs metric or gene.
+++
`cf.pl.violin_box`
:::

:::{grid-item-card} Bar + dot
:link: api/generated/cellfish.plot.bardotplot
:link-type: doc
:img-top: _static/img/gallery/bardotplot.png
:shadow: sm

Mean bars with per-cell points on top.
+++
`cf.pl.bardotplot`
:::

:::{grid-item-card} Ridge
:link: api/generated/cellfish.plot.ridgeplot
:link-type: doc
:img-top: _static/img/gallery/ridgeplot.png
:shadow: sm

Overlapping KDEs of a QC or expression metric.
+++
`cf.pl.ridgeplot`
:::

:::{grid-item-card} plot1cell
:link: api/generated/cellfish.plot.plot1cell
:link-type: doc
:img-top: _static/img/gallery/plot1cell.png
:shadow: sm

Circular UMAP with metadata rings (no R / circlize).
+++
`cf.pl.plot1cell`
:::

:::{grid-item-card} Dotplot
:link: api/generated/cellfish.plot.dotplot
:link-type: doc
:img-top: _static/img/gallery/dotplot.png
:shadow: sm

Marsilea dotplot of mean expression × fraction detected.
+++
`cf.pl.dotplot`
:::

:::{grid-item-card} Palettes
:link: api/generated/cellfish.plot.show_palette
:link-type: doc
:img-top: _static/img/gallery/palettes.png
:shadow: sm

Inspect a dict or named palette before passing ``palette=``.
+++
`cf.pl.show_palette`
:::

::::
