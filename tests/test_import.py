import cellfish as cf


def test_public_namespace():
    assert cf.__all__ == ["pl", "io", "data", "stats", "ext"]
    assert hasattr(cf.io, "write_h5_safe")
    assert hasattr(cf.pl, "setup_style")
    assert hasattr(cf.data, "require_obs")
    assert hasattr(cf.stats, "is_outlier")


def test_ext_lazy_tools():
    assert "drvi" in dir(cf.ext)
    mod = cf.ext.drvi
    assert mod.__name__.endswith("ext.drvi")
