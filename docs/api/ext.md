# Extensions

Each algorithm lives in `cellfish.ext.<tool>` and is imported lazily:

```python
import cellfish as cf
cf.ext.milo  # loaded on first access
```

Optional dependencies belong in `pyproject.toml` extras and should be imported inside functions, not at module import.

Placeholders below are not migrated yet. Implementations will land as `ext/<tool>/_prep.py` + `_plot.py` (or a single `_core.py` until the folder grows).

```{eval-rst}
.. currentmodule:: cellfish.ext

.. autosummary::
    :toctree: generated

    milo
    liana
    archr
    drvi
    scenicplus
    chrombpnet
```
