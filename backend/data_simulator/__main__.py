"""Allow `python -m scripts.data_simulator` to invoke the CLI."""
import sys

from .cli import main

sys.exit(main())
