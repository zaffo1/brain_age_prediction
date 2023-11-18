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
package_name = 'brain_age_prediction'
package_root = os.path.abspath(os.path.join('..','..','..'))
print(package_root)
sys.path.insert(0, package_root)
sys.path.insert(0, os.path.join(package_root, package_name))


# -- Project information -----------------------------------------------------

project = 'brain_age_prediction'
copyright = '2023, Lorenzo Zaffina'
author = 'Lorenzo Zaffina'

# The short X.Y version
from brain_age_prediction.version import __version__
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__



# Path to the URL of the github repository.
base_repo_url = 'https://github.com/zaffo1/brain_age_prediction'

# And here we define some shortcuts.
extlinks = {
    'repourl': ('%s/%%s' % base_repo_url, '[github]/%s')
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc'
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
