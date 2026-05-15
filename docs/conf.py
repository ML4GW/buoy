import sys
from importlib.metadata import distribution
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

dist = distribution("ml4gw-buoy")

project = "buoy"
author = dist.metadata["Author"] or "ML4GW"
copyright = f"2025, {author}"
release = dist.metadata["Version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

autodoc_default_options = {
    "members": None,
    "show-inheritance": None,
}
autodoc_member_order = "bysource"
