"""Allow `python -m scripts.data_simulator` to invoke the CLI."""
import sys

from backend.data_simulator.cli import main

sys.exit(main())
