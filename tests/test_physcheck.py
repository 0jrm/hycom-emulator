"""physcheck_b00 sharpness metrics on synthetic fields."""

import numpy as np

from hycom_emulator.physcheck_b00 import _band, _gradient


def _rms_of_wave(cells):
    x = np.mgrid[:200, :300][1]
    a = np.sin(2 * np.pi * x / cells)
    a[:20, :50] = np.nan
    return np.sqrt(np.nanmean(_band(a)[30:-30, 60:-30] ** 2)) / np.sqrt(0.5)


def test_band_keeps_10_to_50_km_and_drops_the_mesh_scale():
    assert _rms_of_wave(5) > 0.7 and _rms_of_wave(8) > 0.7, "a 20-32 km wave must pass"
    assert _rms_of_wave(60) < 0.05, "a 240 km wave must not"


def test_gradient_sees_a_sharper_front_as_larger():
    x = np.mgrid[:50, :80][1]
    assert np.nanmax(_gradient(np.tanh((x - 40) / 2.0))) > 3 * np.nanmax(_gradient(np.tanh((x - 40) / 8.0)))
