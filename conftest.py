"""Pytest configuration.

Ensures the repository root is on sys.path so tests can import the
`src` package (e.g. `from src.data_generator import _generate_dataset`)
when pytest is run from any working directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
