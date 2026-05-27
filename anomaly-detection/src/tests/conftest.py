"""pytest configuration for anomaly-detection sub-component tests.

Inserts the sub-component root into sys.path so that
    import src.anomaly_detection
resolves correctly without installing the package.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
