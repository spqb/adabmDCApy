# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# this is a trick to make sphinx find the modules in the parent directory
import os
import sys
sys.path.insert(0, os.path.abspath("."))

project = 'adabmDCA'
copyright = '2024, Lorenzo Rosset'
author = 'Lorenzo Rosset, Roberto Netti, Anna Paola Muntoni, Martin Weigt, Francesco Zamponi'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'myst_parser',
	'sphinx.ext.githubpages',
	'sphinx.ext.mathjax',
	'sphinxcontrib.bibtex',
	'sphinxemoji.sphinxemoji',
]

sphinxemoji_source = 'https://unpkg.com/twemoji@latest/dist/twemoji.min.js'

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
]

sphinxemoji_style = 'twemoji'
bibtex_bibfiles = ['bibliography.bib']
#bibtex_encoding = 'latin'
bibtex_reference_style = 'author_year'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_logo = "images/spqb_logo.jpeg"
#html_static_path = ['_static']
