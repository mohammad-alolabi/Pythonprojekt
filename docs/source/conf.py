# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(r"../../"))  # Fügt das Hauptverzeichnis "Pythonprojekt" zum sys.path hinzu


project = 'Neuronale Netzwerke zur Klassifikation von Kugellager-Fehlern'
copyright = '2024, Mohammad Alolabi'
author = 'Mohammad Alolabi'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Falls du Google- oder NumPy-Stil-Dokstrings verwendest
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
]


templates_path = ['_templates']
exclude_patterns = []

language = 'de'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',  # Google Analytics ID
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'black',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# # -- Options for HTML output ----------------------------------------------
# html_theme = 'furo'

# # Theme options are theme-specific and customize the look and feel of a theme further.
# html_theme_options = {
#     "sidebar_hide_name": True,
#     "navigation_with_keys": True,
# }