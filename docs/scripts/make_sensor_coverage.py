"""Render the field-sensor (SQUID + gravimeter) coverage chart.

The magnetometric and gravitational channels sit far below the photon/CR bands
on the shared log10(E/eV) axis (a frequency maps in via E = h*nu), so they get
their own chart: the two instrument bands plus markers for the axion / EP /
oscillating-DM lines their exotic libraries scan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anomalymetric.gravitational.score import grav_exotic_library
from anomalymetric.magnetometric.score import squid_exotic_library
from anomalymetric.units import H_PLANCK_EV_S

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "sensor-coverage.svg"


def _log_e(freq_hz: float) -> float:
    return float(np.log10(H_PLANCK_EV_S * freq_hz))


# (label, f_lo_hz, f_hi_hz, color)
BANDS = [
    ("gravimeter\nµHz–kHz", 1e-6, 1e3, "#a8dadc"),
    ("SQUID\n0.1 Hz–1 GHz", 1e-1, 1e9, "#ffd166"),
]


def _template_center_log_e(tpl) -> float:
    return float(np.log10(tpl.factory().parameters["E_center_eV"].value))


def main() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.6))

    for i, (name, f_lo, f_hi, color) in enumerate(BANDS):
        lo, hi = _log_e(f_lo), _log_e(f_hi)
        y = 0.55 - 0.32 * i
        ax.axvspan(lo, hi, ymin=y - 0.12, ymax=y + 0.12, color=color, alpha=0.5)
        ax.text(0.5 * (lo + hi), y, name, ha="center", va="center", fontsize=9)

    for tpl in squid_exotic_library():
        if tpl.name.startswith(("axion", "darkphoton")):
            ax.axvline(_template_center_log_e(tpl), color="#bc6c25", alpha=0.6, lw=0.8)
    for tpl in grav_exotic_library():
        if tpl.name.startswith(("ep", "oscdm", "yukawa")):
            ax.axvline(_template_center_log_e(tpl), color="#1d3557", alpha=0.6, lw=0.8)

    ax.text(_log_e(1e8), 0.78, "axion / dark-photon lines", color="#bc6c25", fontsize=8, ha="center")
    ax.text(_log_e(1e-2), 0.13, "EP / fifth-force / osc-DM lines", color="#1d3557", fontsize=8, ha="center")

    ax.set_xlim(-20, -4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{eV})$   (sensor frequency via $E=h\nu$)")
    ax.set_title("anomalymetric — field-sensor channel coverage")
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    print(f"wrote {main()}")
