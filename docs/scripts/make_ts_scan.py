"""Animated GIF: TS for each exotic template, computed sequentially."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from anomalymetric.ingest.synthetic import synthetic_natural, synthetic_with_exotic
from anomalymetric.models.exotic import default_library
from anomalymetric.models.mixture import Mixture
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.models.inference import Fit
from anomalymetric.units import wavelength_nm_to_ev

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "ts-scan.gif"


def _natural():
    return [
        BlackBody(T_K=300.0, amplitude=1.0),
        PowerLaw(amplitude=1e-3, index=2.0, reference_eV=1.0),
    ]


def main() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base = synthetic_natural(
        log_e_min=-2, log_e_max=4, bins_per_decade=20,
        T_K=300.0, exposure_cm2_s=1e5, seed=0,
    )
    line_eV = float(wavelength_nm_to_ev(532.0))
    spec = synthetic_with_exotic(base, line_E_eV=line_eV, line_amplitude=2e3, sigma_dex=0.005, seed=1)

    # Fit the natural mixture once.
    nat_fit = Fit(Mixture(_natural(), name="natural"), spec).run()

    lib = default_library()
    names = [t.name for t in lib]
    ts_values: list[float] = [0.0] * len(lib)

    frames = []
    for i, tpl in enumerate(lib):
        alt = Mixture(_natural() + [tpl.factory()], name=f"alt.{tpl.name}")
        alt_fit = Fit(alt, spec).run()
        ts = max(0.0, 2.0 * (alt_fit.log_likelihood - nat_fit.log_likelihood))
        ts_values[i] = ts

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        colors = ["#1d3557"] * len(lib)
        if ts == max(ts_values):
            colors[ts_values.index(max(ts_values))] = "#e63946"
        bars = ax.barh(range(len(lib)), ts_values, color=colors)
        for j in range(i + 1, len(lib)):
            bars[j].set_alpha(0.15)
        ax.set_yticks(range(len(lib)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("test statistic")
        ax.set_title(f"scanning template {i + 1}/{len(lib)}:  {tpl.name}")
        ax.set_xlim(0, max(20.0, max(ts_values) * 1.1))
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    # Hold the final frame longer.
    frames.extend([frames[-1]] * 5)
    imageio.mimsave(OUT_PATH, frames, duration=0.35, loop=0)
    return OUT_PATH


if __name__ == "__main__":
    print(f"wrote {main()}")
