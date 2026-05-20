# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy<2.3",
#     "scipy>=1.14.0",
# ]
# ///
"""
Project-root CLI runner for the welding robot data simulator.

Delegates to backend.data_simulator.cli.

Usage:
    uv run scripts/simulate.py --help
    uv run python scripts/simulate.py --help
"""

import sys
from pathlib import Path

# Add project root to sys.path so backend/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_simulator.cli import main

if __name__ == "__main__":
    sys.exit(main())
