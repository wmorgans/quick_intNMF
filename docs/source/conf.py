master_doc = 'index'

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'sc_intNMF'
copyright = '2022, william morgans'
author = 'william morgans'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.duration',
  'sphinx.ext.doctest',
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary'
]

extensions.append('sphinx.ext.todo')
extensions.append('sphinx.ext.intersphinx')
extensions.append('sphinx.ext.mathjax')
extensions.append('sphinx.ext.viewcode')
extensions.append('sphinx.ext.graphviz')
extensions.append('sphinx.ext.coverage')
extensions.append('sphinx.ext.napoleon')
extensions.append('sphinx_autodoc_typehints')
extensions.append('nbsphinx')
extensions.append('sphinx.ext.napoleon')


# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

autosummary_generate = True
autoclass_content = 'both'
html_show_sourcelink = False
autodoc_inherit_docstrings = True
set_type_checking_flag = True
autodoc_default_flags = ['members']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#    :template: custom_module_template.rst
