import numpy as np
import pytest

from smooth import smooth, window_frames


@pytest.mark.parametrize("fps", [1, 24, 30, 60, 120, None])
def test_window_is_odd_and_longer_than_poly_order(fps):
    window = window_frames(fps)
    assert window % 2 == 1
    assert window >= 5


def test_fills_low_confidence_frames():
    t = np.arange(60)
    xy = np.column_stack([np.full(60, 100.0), 2.0 * t])
    conf_valid = np.ones(60, dtype=bool)
    xy[20:25] = np.nan
    conf_valid[20:25] = False

    out = smooth(xy, conf_valid, fps=30)

    assert np.all(np.isfinite(out))
    #a straight line survives interpolation and a quadratic savgol filter unchanged
    np.testing.assert_allclose(out[:, 1], 2.0 * t, atol=1e-3)


def test_too_short_to_smooth_returns_input():
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert smooth(xy, np.array([True, True]), fps=30) is xy
