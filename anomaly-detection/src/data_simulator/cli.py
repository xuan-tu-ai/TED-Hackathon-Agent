"""
Command-line interface for the welding robot data simulator.

Usage:
    uv run python -m scripts.data_simulator.cli --help
    uv run python -m scripts.data_simulator.cli --train --test --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .batch_generator import BatchGenerator
from .writer import CSVWriter
from .config import SimulatorConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Physics-based welding robot data simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full 30-day train + test dataset
  uv run python -m scripts.data_simulator.cli

  # Generate only train data with custom robots
  uv run python -m scripts.data_simulator.cli --train --no-test --robots 0 1 2

  # Generate test data with specific anomaly ratio
  uv run python -m scripts.data_simulator.cli --no-train --test --test-cycles 500 --anomaly-ratio 0.3

  # Custom output directory
  uv run python -m scripts.data_simulator.cli --output-dir ./my_data
        """,
    )
    parser.add_argument(
        "--train", action="store_true", default=True,
        dest="train", help="Generate training dataset (default: True)",
    )
    parser.add_argument(
        "--no-train", action="store_false", dest="train",
        help="Skip training dataset generation",
    )
    parser.add_argument(
        "--test", action="store_true", default=True,
        dest="test", help="Generate test dataset (default: True)",
    )
    parser.add_argument(
        "--no-test", action="store_false", dest="test",
        help="Skip test dataset generation",
    )
    parser.add_argument(
        "--num-days", type=int, default=30,
        help="Number of days for training data (default: 30)",
    )
    parser.add_argument(
        "--test-cycles", type=int, default=2000,
        help="Number of test cycles (default: 2000)",
    )
    parser.add_argument(
        "--anomaly-ratio", type=float, default=0.5,
        help="Fraction of test cycles that are anomalous (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/generated",
        help="Output directory for CSV files (default: data/generated)",
    )
    parser.add_argument(
        "--robots", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Robot IDs to generate data for (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000,
        help="Rows per CSV write chunk (default: 10000)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args(argv)


def print_summary(
    train_path: str | None,
    train_count: int,
    test_path: str | None,
    test_count: int,
    label_path: str | None,
    elapsed: float,
    robots: list[int],
    num_days: int,
) -> None:
    """Print a summary of the generated dataset."""
    print("\n" + "=" * 56)
    print("  WELDING ROBOT DATA SIMULATOR — COMPLETE")
    print("=" * 56)
    print(f"  Robots:          {robots}")
    print(f"  Days:            {num_days}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print()

    if train_path:
        train_mb = os.path.getsize(train_path) / (1024 * 1024)
        print(f"  Train dataset:")
        print(f"    File:    {train_path}")
        print(f"    Rows:    {train_count:,}")
        print(f"    Size:    {train_mb:.1f} MB")

    if test_path:
        test_mb = os.path.getsize(test_path) / (1024 * 1024)
        print(f"  Test dataset:")
        print(f"    File:    {test_path}")
        print(f"    Rows:    {test_count:,}")
        print(f"    Size:    {test_mb:.1f} MB")

    if label_path:
        print(f"  Labels:  {label_path}")

    print("=" * 56)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the data simulator CLI."""
    args = parse_args(argv)
    start_time = time.time()

    if not args.train and not args.test:
        print("Error: Nothing to do. Use --train and/or --test.")
        return 1

    config = SimulatorConfig(
        seed=args.seed,
        num_days=args.num_days,
        robots=args.robots,
    )

    # Sprint 18: Use training_mode=True for convergent normal data
    # (reduced noise + deterministic per (robot, program) generation)
    train_batch_gen = BatchGenerator(seed=config.seed, training_mode=True)
    test_batch_gen = BatchGenerator(seed=config.seed)

    writer = CSVWriter(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )

    train_path = None
    train_count = 0
    test_path = None
    test_count = 0
    label_path = None

    # Generate training data
    if args.train:
        if not args.quiet:
            print(f"Generating training data ({args.num_days} days, robots={args.robots})...")
            sys.stdout.flush()

        t0 = time.time()
        train_path, train_count = writer.write_train(
            train_batch_gen,
            num_days=args.num_days,
            robots=args.robots,
        )
        if not args.quiet:
            t = time.time() - t0
            print(f"  Done: {train_count:,} rows in {t:.1f}s")

    # Generate test data
    if args.test:
        # Re-seed for test to ensure independent randomness
        test_seed = args.seed + 1000
        if not args.quiet:
            print(f"Generating test data ({args.test_cycles} cycles, {args.anomaly_ratio:.0%} anomaly)...")
            sys.stdout.flush()

        t0 = time.time()
        test_path, label_path, test_count = writer.write_test(
            test_batch_gen,
            num_cycles=args.test_cycles,
            anomaly_ratio=args.anomaly_ratio,
            robots=args.robots,
            seed=test_seed,
        )
        if not args.quiet:
            t = time.time() - t0
            print(f"  Done: {test_count:,} rows in {t:.1f}s")

    # Print summary
    if not args.quiet:
        elapsed = time.time() - start_time
        print_summary(
            train_path, train_count,
            test_path, test_count,
            label_path, elapsed,
            args.robots, args.num_days,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
