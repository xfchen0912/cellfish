import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellfish.data import require_layer, require_obs, require_obsm, require_var


def _tiny_adata() -> AnnData:
    adata = AnnData(np.zeros((4, 3)))
    adata.obs_names = [f"c{i}" for i in range(4)]
    adata.var_names = ["g1", "g2", "g3"]
    adata.obs["cell_type"] = ["A", "A", "B", "B"]
    adata.obsm["X_umap"] = np.zeros((4, 2))
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_require_obs_ok():
    require_obs(_tiny_adata(), "cell_type")


def test_require_obs_missing():
    with pytest.raises(KeyError, match="sample"):
        require_obs(_tiny_adata(), ["cell_type", "sample"])


def test_require_obsm_alias():
    adata = _tiny_adata()
    assert require_obsm(adata, "umap") == "X_umap"
    assert require_obsm(adata, "X_umap") == "X_umap"


def test_require_var_and_layer():
    adata = _tiny_adata()
    require_var(adata, ["g1", "g2"])
    require_layer(adata, "counts")
    with pytest.raises(KeyError):
        require_var(adata, "missing")
    with pytest.raises(KeyError):
        require_layer(adata, "scaled")


def test_pairing():
    from cellfish.data import paired_rna_atac_labels

    rna = pd.Series(["Hep", "T"], index=["rna1", "rna2"])
    atac = pd.DataFrame(
        {"cell_type_highres": ["HepA", "TA"], "RNA_paired_cell": ["rna1", "rna2"]},
        index=["atac1", "atac2"],
    )
    out = paired_rna_atac_labels(rna, atac)
    assert list(out.columns) == ["atac_label", "rna_label"]
    assert list(out["rna_label"]) == ["Hep", "T"]
