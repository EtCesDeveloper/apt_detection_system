"""
Environment setup — silences noisy TF logs.
Import this BEFORE importing tensorflow anywhere in the project.

The src package imports it automatically via __init__.py, so you don't
need to call it manually unless you write standalone scripts.
"""

import os
import warnings

# Suppress TF C++ logs (0=all, 1=INFO off, 2=WARNING off, 3=ERROR only)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Suppress oneDNN message (we keep oneDNN ON for the perf boost,
# we only hide the informational log line)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

# Hide deprecation warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
