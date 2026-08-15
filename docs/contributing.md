# Contributing

This is a personal helpers package. Keep the public surface small: `cf.io`, `cf.data`, `cf.pl`, `cf.stats`, `cf.ext`.

Scanpy’s [developer documentation](https://scanpy.readthedocs.io/en/latest/dev/index.html) covers git, tests, and docstrings. The notes below are cellfish-specific.

## Install

```bash
cd cellfish
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot,doc]"
```

Tests:

```bash
pytest
```

## Adding an algorithm

1. Create `src/cellfish/ext/<tool>/__init__.py` plus one implementation file.
2. Keep compute and plotting together; reuse `cf.pl` for fonts, palettes, and grids.
3. Put optional dependencies in `pyproject.toml` extras and import them inside functions.
4. Add a smoke test with a tiny simulated object.
5. Do not add `plot/_<tool>.py`, and do not change layer 1 unless two tools share the logic.

Layer 1 (`io` / `data` / `plot` / `stats`) must not import `ext`.

## Documentation

This project uses [Sphinx](https://www.sphinx-doc.org/) with MyST markdown, Napoleon docstrings, and autosummary — the same layout as scMagnify.

- New public functions need a docstring (numpy or google style) and an entry in `docs/api/`.
- Tutorials live in `docs/tutorials/` and `docs/notebooks/`.
- Citations use `{cite:p}`Key`` once `docs/references.bib` has the entry.

If you refer to objects from other packages, add them to `intersphinx_mapping` in `docs/conf.py`.
If a missing link is outside your control, add it to `nitpick_ignore`.

### Build locally

```bash
pip install -e ".[doc,plot]"
cd docs
make html
python -m webbrowser -t _build/html/index.html
```

Generated autosummary pages land in `docs/api/generated/` and are not committed.

## Code style

Optional [pre-commit](https://pre-commit.com/) and ruff are listed in `pyproject.toml`. Do not expand layer 1 to absorb paper-specific palettes or figure panels.
