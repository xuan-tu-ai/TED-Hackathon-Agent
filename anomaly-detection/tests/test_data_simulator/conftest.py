"""Configure test path so that backend.data_simulator is importable.

Since tests are now inside the backend/ package, backend.data_simulator
is available through the standard Python path (backend/ is on sys.path
and now has __init__.py).
"""

