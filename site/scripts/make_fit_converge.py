"""Animated GIF: Nelder-Mead iterations of a blackbody fit converging onto data.

Uses scipy.optimize.minimize directly with a callback so we don't need to
modify the library's `Fit` API.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from anomalymetric.ingest.synthetic import synthetic_natural
from anomalymetric.models.inference import (
    observed_counts_from_spectrum,
    poisson_log_likelihood,
)
from anomalymetric.models.thermal import BlackBody

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "fit-converge.gif"


def main() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    spec = synthetic_natural(
        log_e_min=-2, log_e_max=2, bins_per_decade=40,
        T_K=300.0, bb_amplitude=1.0, pl_amplitude=1e-8,
        exposure_cm2_s=1e10, poisson=True, seed=42,
    )
    centers = spec.energy_centers_eV
    log_centers = np.log10(centers)
    widths = spec.bin_widths_eV
    observed = observed_counts_from_spectrum(spec)

    model = BlackBody(T_K=150.0, amplitude=0.3)
    history: list[tuple[float, float, float]] = []

    def neg_ll(x: np.ndarray) -> float:
        model.parameters["T_K"].value = float(np.clip(x[0], 1.0, 1e6))
        model.parameters["amplitude"].value = float(max(x[1], 1e-12))
        mu = model.dnde(log_centers) * widths
        ll = poisson_log_likelihood(observed, mu)
        return -ll

    def cb(xk):
        history.append((float(xk[0]), float(xk[1]), -float(neg_ll(xk))))

    cb(np.array([150.0, 0.3]))
    minimize(
        neg_ll,
        [150.0, 0.3],
        method="Nelder-Mead",
        callback=cb,
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400},
    )

    # Subsample to ≤ 30 frames so the GIF stays slim.
    step = max(1, len(history) // 30)
    history = history[::step]

    ymax = observed.max() * 3
    ymin = max(1.0, observed[observed > 0].min() * 0.5) if (observed > 0).any() else 1.0

    frames = []
    for i, (T, A, ll) in enumerate(history):
        model.parameters["T_K"].value = T
        model.parameters["amplitude"].value = A
        mu = model.dnde(log_centers) * widths
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=110)
        ax.step(centers, observed, where="mid", color="#1d3557", label="observed")
        ax.plot(centers, mu, color="#e63946", lw=2, label=f"BB(T={T:.0f}, A={A:.2g})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("E [eV]")
        ax.set_ylabel("counts")
        ax.set_title(f"Nelder-Mead step {i + 1}/{len(history)}   log L = {ll:.2f}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    frames.extend([frames[-1]] * 5)
    imageio.mimsave(OUT_PATH, frames, duration=0.15, loop=0)
    return OUT_PATH


if __name__ == "__main__":
    print(f"wrote {main()}")
