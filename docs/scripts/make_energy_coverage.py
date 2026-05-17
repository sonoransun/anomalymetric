"""Render the radio→ZeV coverage chart.

Bands show the canonical electromagnetic regimes; markers show where each
exotic template in `default_library()` lives; shaded tracks indicate where the
two natural-mixture components (BlackBody @ 300 K, generic power-law) carry
appreciable flux.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anomalymetric.models.exotic import LASER_LINES_NM, default_library
from anomalymetric.models.thermal import BlackBody
from anomalymetric.units import wavelength_nm_to_ev

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "energy-coverage.svg"

EM_BANDS = [
    ("radio", -7, -3, "#8ecae6"),
    ("μwave", -3, -2, "#a8dadc"),
    ("IR", -2, 0.3, "#bde0fe"),
    ("vis", 0.3, 0.6, "#ffd166"),
    ("UV", 0.6, 2, "#cdb4db"),
    ("X-ray", 2, 5, "#ffadad"),
    ("γ", 5, 11, "#ff6b6b"),
    ("TeV", 11, 14, "#ff476f"),
    ("PeV", 14, 17, "#d00000"),
    ("EeV", 17, 19, "#9d0208"),
    ("ZeV", 19, 21, "#370617"),
]


def main() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))

    for name, lo, hi, color in EM_BANDS:
        ax.axvspan(lo, hi, color=color, alpha=0.35)
        ax.text(
            0.5 * (lo + hi),
            0.96,
            name,
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

    # Natural components: blackbody (T=300K) and a power-law tail.
    log_E = np.linspace(-6.5, 21, 1200)
    bb = BlackBody(T_K=300.0, amplitude=1.0).dnde(log_E)
    bb = bb / bb.max()
    pl = 10.0 ** (-(log_E + 6.5) * 0.18)
    pl = pl / pl.max()
    ax.plot(log_E, bb * 0.55 + 0.06, color="#1d3557", lw=2, label="Blackbody 300 K")
    ax.plot(log_E, pl * 0.55 + 0.06, color="#2a9d8f", lw=2, ls="--", label="Astro power-law")

    # Exotic-template markers
    for label, lam in LASER_LINES_NM.items():
        E_ev = float(wavelength_nm_to_ev(lam))
        ax.axvline(np.log10(E_ev), color="#264653", alpha=0.5, lw=0.8)
        ax.text(np.log10(E_ev), 0.02, "laser", rotation=90, fontsize=7, ha="right", va="bottom")
    for tpl in default_library():
        # Mark hard cutoffs and GZK by template name parsing — keep simple.
        name = tpl.name
        if "hardcut" in name:
            mapping = {"hardcut.100keV": 5, "hardcut.1TeV": 12, "hardcut.1PeV": 15}
            x = mapping.get(name)
            if x is not None:
                ax.scatter([x], [0.66], marker="v", color="#bc4749", s=60, zorder=5)
                ax.text(x, 0.69, "cutoff", fontsize=7, ha="center", va="bottom")
        if "gzk" in name:
            ax.scatter([np.log10(5e19)], [0.66], marker="X", color="#9d0208", s=90, zorder=5)
            ax.text(np.log10(5e19), 0.69, "GZK tail", fontsize=7, ha="center", va="bottom")

    ax.set_xlim(-7, 21)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{eV})$")
    ax.set_ylabel("relative dN/dE  (schematic)")
    ax.set_title("anomalymetric — energy coverage")
    ax.set_yticks([])
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    print(f"wrote {main()}")
