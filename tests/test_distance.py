"""Workstream C coverage: forward/distance.py redshift + K-correction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from anomalymetric.forward.distance import (
    k_correction_factor,
    luminosity_distance_cm,
    shift_to_rest_frame,
)


def test_shift_to_rest_frame_adds_log1pz():
    edges = np.array([0.0, 1.0, 2.0])
    z = 1.0
    shifted = shift_to_rest_frame(edges, z)
    assert np.allclose(shifted, edges + np.log10(2.0))
    # z=0 is a no-op.
    assert np.allclose(shift_to_rest_frame(edges, 0.0), edges)


def test_shift_to_rest_frame_rejects_negative_z():
    with pytest.raises(ValueError):
        shift_to_rest_frame(np.array([0.0, 1.0]), -0.5)


def test_k_correction_factor():
    # alpha=1 -> factor 1 for any z; alpha=2, z=1 -> (1+z)^1 = 2.
    assert k_correction_factor(1.0, 3.0) == pytest.approx(1.0)
    assert k_correction_factor(2.0, 1.0) == pytest.approx(2.0)


def test_luminosity_distance_increases_with_z():
    d1 = luminosity_distance_cm(0.1)
    d2 = luminosity_distance_cm(1.0)
    assert d2 > d1 > 0.0
