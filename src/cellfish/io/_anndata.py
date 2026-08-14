"""Compatibility helpers for writing AnnData and MuData objects."""

from __future__ import annotations

from typing import Literal, Union, cast

import pandas as pd
from anndata import AnnData
from pandas.api.types import is_object_dtype, is_string_dtype

try:
    from mudata import MuData
except ImportError:  # pragma: no cover
    MuData = None  # type: ignore[assignment]


DataObject = Union[AnnData, "MuData"]
CopyMode = Literal["inplace", "copy"]


def _sanitize_index(index: pd.Index) -> pd.Index:
    values = [None if pd.isna(v) else str(v) for v in index]
    return pd.Index(values, dtype="object", name=index.name)


def _coerce_object_series(series: pd.Series, numeric_threshold: float) -> pd.Series:
    non_na = int(series.notna().sum())
    if non_na == 0:
        return series.astype("object")

    as_num = pd.to_numeric(series, errors="coerce")
    converted = int(as_num.notna().sum())
    if converted / non_na >= numeric_threshold:
        return as_num

    return series.map(lambda x: None if pd.isna(x) else str(x)).astype("object")


def _sanitize_dataframe(df: pd.DataFrame, numeric_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out.index = _sanitize_index(out.index)

    for col in out.columns:
        series = out[col]
        dtype = series.dtype

        if isinstance(dtype, pd.CategoricalDtype):
            out[col] = series.astype("object")
            continue

        if is_string_dtype(dtype):
            out[col] = series.astype("object").where(series.notna(), None)
            continue

        if is_object_dtype(dtype):
            out[col] = _coerce_object_series(series, numeric_threshold=numeric_threshold)

    return out


def _sanitize_adata(adata: AnnData, numeric_threshold: float) -> AnnData:
    adata.obs = _sanitize_dataframe(adata.obs, numeric_threshold=numeric_threshold)
    adata.var = _sanitize_dataframe(adata.var, numeric_threshold=numeric_threshold)
    adata.obs_names = _sanitize_index(adata.obs_names)
    adata.var_names = _sanitize_index(adata.var_names)
    return adata


def sanitize_for_h5_write(
    obj: DataObject,
    *,
    mode: CopyMode = "inplace",
    numeric_threshold: float = 0.95,
) -> DataObject:
    """Normalize metadata dtypes to improve h5ad/h5mu writing compatibility."""
    if mode not in {"inplace", "copy"}:
        raise ValueError("mode must be either 'inplace' or 'copy'.")

    if not (0.0 <= numeric_threshold <= 1.0):
        raise ValueError("numeric_threshold must be in [0.0, 1.0].")

    target = obj if mode == "inplace" else obj.copy()

    if isinstance(target, AnnData):
        return _sanitize_adata(target, numeric_threshold=numeric_threshold)

    if MuData is not None and isinstance(target, MuData):
        target.obs = _sanitize_dataframe(target.obs, numeric_threshold=numeric_threshold)
        target.var = _sanitize_dataframe(target.var, numeric_threshold=numeric_threshold)

        for mod_name in list(target.mod.keys()):
            target.mod[mod_name] = _sanitize_adata(
                target.mod[mod_name], numeric_threshold=numeric_threshold
            )

        return target

    raise TypeError("obj must be an AnnData or MuData instance.")


def write_h5_safe(
    obj: DataObject,
    filename: str,
    *,
    mode: CopyMode = "inplace",
    numeric_threshold: float = 0.95,
    **write_kwargs,
) -> DataObject:
    """Sanitize metadata and write AnnData/MuData to disk.

    For :class:`AnnData`, this calls ``.write_h5ad``.
    For :class:`MuData`, this calls ``.write`` (h5mu output path).
    """
    sanitized = sanitize_for_h5_write(
        obj,
        mode=mode,
        numeric_threshold=numeric_threshold,
    )

    if isinstance(sanitized, AnnData):
        sanitized.write_h5ad(filename, **write_kwargs)
        return sanitized

    if MuData is not None and isinstance(sanitized, MuData):
        cast("MuData", sanitized).write(filename, **write_kwargs)
        return sanitized

    raise TypeError("obj must be an AnnData or MuData instance.")
