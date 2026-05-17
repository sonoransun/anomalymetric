# Generated figures

Every file here is produced by a script under `docs/scripts/`. Regenerate the
full set with:

```
.venv/bin/python docs/scripts/make_all.py
```

| File | Producer | Notes |
| --- | --- | --- |
| `energy-coverage.svg` | `make_energy_coverage.py` | Static SVG of the radio→ZeV log-energy axis with model + template coverage. |
| `pipeline-overview.svg` | `make_pipeline_svg.py` | Hand-authored animated SVG (CSS keyframes) of the end-to-end flow. |
| `planck-sweep.gif` | `make_planck_sweep.py` | Planck dN/dE as T sweeps 100 K → 10 000 K. |
| `line-injection.gif` | `make_line_injection.py` | Background + injected 532 nm line growing; live TS bar. |
| `fit-converge.gif` | `make_fit_converge.py` | Nelder-Mead iterations of a blackbody fit converging onto data. |
| `ts-scan.gif` | `make_ts_scan.py` | Per-template TS bar chart scanning the default exotic library. |

`make_line_injection.py` is the slowest (it runs a full Loeb–Turner score per
frame); the others complete in <10 s on a laptop.
