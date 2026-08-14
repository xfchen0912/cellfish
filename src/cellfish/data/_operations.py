"""AnnData table helpers. Feature names are not assumed to be genes."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd
from anndata import AnnData


def filter_features(adata: AnnData, features: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Keep groups whose names are all present in ``adata.var_names``."""
    return {k: v for k, v in features.items() if all(g in adata.var_names for g in v)}


def map_label_to_cluster(
    mapping_dict: Dict[str, List[str]], max_cluster: Optional[int] = None
) -> Dict[str, str]:
    """Map cluster numbers to labels. Keys are cell types, values are cluster ids."""
    cluster_label: Dict[str, str] = {}
    for labels, clusters in mapping_dict.items():
        for cluster in clusters:
            cluster_label[str(cluster)] = labels

    if max_cluster is None:
        max_cluster = max(int(cluster) for cluster in cluster_label)

    missing_clusters = [
        str(cluster) for cluster in range(max_cluster + 1) if str(cluster) not in cluster_label
    ]
    if missing_clusters:
        print(f"Warning: The following cluster numbers are missing in the mapping: {missing_clusters}")

    return cluster_label


def get_markers(
    adata: AnnData,
    groupby: str,
    key: str = "rank_genes_groups",
    p_val_cutoff: float = 0.05,
    logfc_cutoff: float = 0.5,
) -> pd.DataFrame:
    """Extract marker table from ``adata.uns[key]`` (scanpy rank_genes_groups)."""
    del groupby  # grouping is already encoded in uns[key]
    markers = pd.concat(
        [
            pd.DataFrame(adata.uns[key]["names"]).melt(),
            pd.DataFrame(adata.uns[key]["pvals_adj"]).melt(),
            pd.DataFrame(adata.uns[key]["logfoldchanges"]).melt(),
            pd.DataFrame(adata.uns[key]["scores"]).melt(),
        ],
        axis=1,
    )
    markers.columns = ("cluster", "gene", "cluster2", "p_val_adj", "cluster3", "avg_logFC", "cluster4", "score")
    markers = markers.loc[:, ["cluster", "gene", "avg_logFC", "p_val_adj", "score"]]
    markers = markers.loc[markers.avg_logFC > logfc_cutoff, :]
    markers = markers.loc[markers.p_val_adj < p_val_cutoff, :]
    return markers


def replace_prefix(index: pd.Index, old_prefix: str, new_prefix: str) -> pd.Index:
    """Replace a prefix in index values."""
    return index.map(lambda x: re.sub(rf"^{re.escape(old_prefix)}", new_prefix, str(x)))


def replace_suffix(index: pd.Index, old_suffix: str, new_suffix: str) -> pd.Index:
    """Replace a suffix in index values."""
    return index.map(lambda x: re.sub(rf"{re.escape(old_suffix)}$", new_suffix, str(x)))
