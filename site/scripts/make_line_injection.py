"""Animated GIF: 532 nm laser line growing on a blackbody background, with live TS bar."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from anomalymetric.ingest.synthetic import synthetic_natural, synthetic_with_exotic
from anomalymetric.models.powerlaw import PowerLaw
from anomalymetric.models.thermal import BlackBody
from anomalymetric.score.loeb_turner import loeb_turner_score
from anomalymetric.units import wavelength_nm_to_ev

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "line-injection.gif"
N_FRAMES = 20
LINE_EV = float(wavelength_nm_to_ev(532.0))


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
    amplitudes = np.linspace(0.0, 5e3, N_FRAMES)
    ts_history: list[float] = []
    frames = []

    centers = base.energy_centers_eV
    ymax = base.value.max() * 5

    for amp in amplitudes:
        if amp == 0:
            spec = base
        else:
            spec = synthetic_with_exotic(
                base, line_E_eV=LINE_EV, line_amplitude=float(amp), sigma_dex=0.005, seed=1
            )
        res = loeb_turner_score(spec, _natural())
        ts_history.append(res.test_statistic)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), dpi=110, gridspec_kw={"width_ratios": [3, 1]})
        axL.step(centers, spec.value, where="mid", color="#1d3557")
        axL.axvline(LINE_EV, color="#e63946", ls="--", lw=1, alpha=0.7, label="532 nm")
        axL.set_xscale("log")
        axL.set_yscale("log")
        axL.set_ylim(0.5, ymax)
        axL.set_xlabel("E [eV]")
        axL.set_ylabel("counts")
        axL.set_title(f"injected amplitude = {amp:.0f}")
        axL.legend(loc="upper right")
        axR.barh(["TS"], [res.test_statistic], color="#e63946")
        axR.set_xlim(0, max(50, max(ts_history) * 1.1))
        axR.set_title("test statistic")
        axR.set_xlabel("TS")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(OUT_PATH, frames, duration=0.18, loop=0)
    return OUT_PATH


if __name__ == "__main__":
    print(f"wrote {main()}")
