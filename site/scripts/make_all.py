"""Regenerate every figure and animation in docs/img/.

Each `make_*.py` script is standalone-runnable; this driver imports them by
file path and invokes their `main()` so a single `python make_all.py` produces
the full set.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

JOB_NAMES = [
    "make_energy_coverage",
    "make_sensor_coverage",
    "make_pipeline_svg",
    "make_planck_sweep",
    "make_ts_scan",
    "make_line_injection",
    "make_fit_converge",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    for name in JOB_NAMES:
        t0 = time.time()
        path = _load(name).main()
        dt = time.time() - t0
        size_kb = path.stat().st_size / 1024.0
        print(f"  {name:<22s} {size_kb:8.1f} KB  {dt:6.2f} s  -> {path}")


if __name__ == "__main__":
    main()
