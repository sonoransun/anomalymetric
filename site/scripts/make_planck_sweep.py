"""Animated GIF: BlackBody dN/dE as temperature sweeps 100 K → 10000 K."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from anomalymetric.models.thermal import BlackBody

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "planck-sweep.gif"
N_FRAMES = 40


def main() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_E = np.linspace(-3.5, 4.5, 600)
    temps = np.logspace(np.log10(100.0), np.log10(10000.0), N_FRAMES)

    frames = []
    for T in temps:
        bb = BlackBody(T_K=float(T), amplitude=1.0).dnde(log_E)
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=110)
        ax.plot(log_E, bb, color="#1d3557", lw=2)
        ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{eV})$")
        ax.set_ylabel(r"$dN/dE$  (arb.)")
        ax.set_yscale("log")
        ax.set_xlim(log_E[0], log_E[-1])
        ax.set_ylim(1e-6, 1e6)
        ax.axvspan(np.log10(1.65), np.log10(3.26), color="#ffd166", alpha=0.25, label="visible")
        ax.set_title(f"Planck blackbody  T = {T:.0f} K")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(OUT_PATH, frames, duration=0.1, loop=0)
    return OUT_PATH


if __name__ == "__main__":
    print(f"wrote {main()}")
