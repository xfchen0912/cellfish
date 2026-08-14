"""Join tables across modalities. Plotting stays in ``cellfish.plot``."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

_MISSING = {"nan", "None", "", "NA", "<NA>"}


def join_labels(
    left_labels: pd.Series,
    right_obs: pd.DataFrame,
    *,
    right_label_col: str,
    pair_col: str,
    left_name: str = "left_label",
    right_name: str = "right_label",
) -> pd.DataFrame:
    """Join ``right_obs[pair_col]`` to ``left_labels`` by index.

    Returns a DataFrame indexed like ``right_obs``, with columns
    ``right_name`` and ``left_name``.
    """
    if right_label_col not in right_obs.columns:
        raise KeyError(f"{right_label_col!r} not in right_obs")
    if pair_col not in right_obs.columns:
        raise KeyError(f"{pair_col!r} not in right_obs")

    paired = right_obs[[right_label_col, pair_col]].copy()
    paired[pair_col] = paired[pair_col].astype(str)
    paired = paired[~paired[pair_col].isin(_MISSING)]
    left_index = pd.Index(left_labels.index.astype(str))
    paired = paired[paired[pair_col].isin(left_index)]
    if paired.empty:
        raise ValueError(f"No rows with a resolvable {pair_col}")

    out = pd.DataFrame(
        {
            right_name: paired[right_label_col].astype(str).values,
            left_name: left_labels.astype(str).reindex(paired[pair_col]).values,
        },
        index=paired.index,
    )
    return out.dropna()


def paired_rna_atac_labels(
    rna_labels: pd.Series,
    atac_obs: pd.DataFrame,
    *,
    atac_label_col: str = "cell_type_highres",
    rna_pair_col: str = "RNA_paired_cell",
) -> pd.DataFrame:
    """Join ATAC cells to RNA labels via ``RNA_paired_cell``.

    Returns columns ``atac_label``, ``rna_label`` (one row per paired ATAC cell).
    """
    return join_labels(
        rna_labels,
        atac_obs,
        right_label_col=atac_label_col,
        pair_col=rna_pair_col,
        left_name="rna_label",
        right_name="atac_label",
    )


def correspondence_counts(
    paired: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    row_order: Sequence[str] | None = None,
    col_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Contingency table from a paired label DataFrame."""
    ct = pd.crosstab(paired[row_col], paired[col_col])
    if row_order is not None:
        rows = [r for r in row_order if r in ct.index] + [r for r in ct.index if r not in row_order]
        ct = ct.reindex(index=rows, fill_value=0)
    if col_order is not None:
        cols = [c for c in col_order if c in ct.columns] + [c for c in ct.columns if c not in col_order]
        ct = ct.reindex(columns=cols, fill_value=0)
    return ct
