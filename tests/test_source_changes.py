"""datastore.kept_windows: train samples that straddle a source-experiment change. Synthetic time axes."""

import numpy as np

from hycom_emulator.datastore import SOURCE_CHANGES, kept_windows


def _days(first, n):
    """Daily axis stamped at 12:00, as the pack's time axis is."""
    return np.datetime64(f"{first}T12:00", "ns") + np.arange(n) * np.timedelta64(1, "D")


def test_windows_across_the_2017_splice_are_dropped():
    t = _days("2017-05-28", 10)
    # windows of 3 rows: [05-30, 06-01] ends on a change, [06-01, 06-03] holds 06-02, [06-02, 06-04] starts on one
    assert kept_windows(t, 3, 7).tolist() == [0, 1, 5, 6]


def test_windows_across_2021_01_01_are_dropped_and_n_samples_caps_the_indices():
    t = _days("2020-12-28", 9)
    assert kept_windows(t, 4, 6).tolist() == [0, 4, 5]
    assert kept_windows(t, 4, 5).tolist() == [0, 4]


def test_no_change_inside_the_axis_keeps_everything():
    assert kept_windows(_days("2005-03-01", 30), 6, 25).tolist() == list(range(25))
    assert kept_windows(_days("2021-01-01", 10), 4, 7).tolist() == list(range(7)), "a change on the first row is no straddle"


def test_the_change_list_is_sorted_whole_days():
    assert SOURCE_CHANGES.dtype == np.dtype("datetime64[D]") and (np.diff(SOURCE_CHANGES) > np.timedelta64(0, "D")).all()
