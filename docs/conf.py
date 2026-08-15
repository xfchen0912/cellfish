# Configuration file for the Sphinx documentation builder.
# Template adapted from scMagnify (docs/conf.py).

from importlib.metadata import metadata
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

try:
    info = metadata("cellfish")
    project_name = info.get("Name", "cellfish")
    author = info.get("Author", "Xufeng Chen")
    version = info.get("Version", "0.1.0")
    urls = dict(pu.split(", ") for pu in (info.get_all("Project-URL") or []))
    repository_url = urls.get("Source", "https://github.com/xfchen0912/cellfish")
except Exception:
    project_name = "cellfish"
    author = "Xufeng Chen"
    version = "0.1.0"
    repository_url = "https://github.com/xfchen0912/cellfish"

release = version

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = False
needs_sphinx = "4.0"

html_context = {
    "display_github": True,
    "github_user": "xfchen0912",
    "github_repo": "cellfish",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
autodoc_mock_imports = []
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "attrs_inline",
]
myst_fence_as_directive = ["mermaid"]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mudata": ("https://mudata.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "notebooks/README.md",
    "_gallery_cards.md",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/img/cellfish_icon.png"
html_css_files = ["css/custom.css"]
html_title = project_name

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    ("py:class", "anndata.AnnData"),
    ("py:class", "anndata._core.anndata.AnnData"),
    ("py:class", "matplotlib.axes._axes.Axes"),
    ("py:class", "matplotlib.figure.Figure"),
    ("py:class", "matplotlib.colors.Colormap"),
    ("py:class", "matplotlib.colors.Normalize"),
    ("py:class", "cycler.Cycler"),
]
