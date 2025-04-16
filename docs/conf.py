import os
import sys
sys.path.insert(0, os.path.abspath('../dataria'))

extensions = [
    'myst_parser',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../dataria']

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'

source_suffix = {
    '.md': 'markdown',
}