"""Verify that every .py file under target directories has a module-level docstring.

Usage:
    uv run python scripts/check_docstrings.py
"""

import ast
import sys
from pathlib import Path


TARGET_DIRS = [
    Path("backend/app/anomaly_detection"),
    Path("backend/app/robot_agent"),
    Path("backend/agent_knowledge"),
]


def has_module_docstring(filepath: Path) -> bool:
    """Return True if the file's first statement (after shebang/encoding) is a
    string expression, i.e. a module-level docstring."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    if not tree.body:
        return False

    first = tree.body[0]
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)


def main() -> int:
    missing: list[Path] = []
    checked = 0

    for d in TARGET_DIRS:
        if not d.is_dir():
            print(f"[SKIP] Directory not found: {d}")
            continue
        for py_file in sorted(d.rglob("*.py")):
            checked += 1
            if not has_module_docstring(py_file):
                missing.append(py_file)

    print(f"\nChecked {checked} .py files.\n")

    if not missing:
        print("All files have a module-level docstring.")
        return 0

    print(f"Files MISSING module-level docstrings ({len(missing)}):\n")
    for f in missing:
        print(f"  {f}")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
