# Gallery

Card catalog of layer-1 plots. Each tile opens the corresponding {doc}`api/index` page.
Thumbnails use the full epithelial snRNA AnnData (all cells; marker genes only
in memory), with cluster colors passed in via ``palette=``. Card images are
padded to a 4:3 landscape frame.

Regenerate images from the repository root:

```bash
python docs/scripts/make_gallery.py
```

```{include} _gallery_cards.md
```
