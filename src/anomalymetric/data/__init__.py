"""Packaged reference data + a small accessor.

Reference tables live in ``anomalymetric/data/references/`` and are declared as
package-data in ``pyproject.toml`` so they ship with the wheel. Use
``load_reference(name)`` to read one as an ``astropy.table.Table``.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Any


def reference_names() -> list[str]:
    """List the available reference-data files."""
    root = files("anomalymetric.data.references")
    return sorted(p.name for p in root.iterdir() if p.name.endswith((".csv", ".dat")))


def load_reference(name: str) -> Any:
    """Load a packaged reference table by file name as an `astropy.table.Table`."""
    from astropy.table import Table

    resource = files("anomalymetric.data.references").joinpath(name)
    with as_file(resource) as path:
        return Table.read(str(path), format="ascii.csv")
