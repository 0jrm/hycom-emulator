import inspect

import numpy as np
import pytest

from hycom_emulator.forcing_mode import FORCING_WINDOW, Forcing, Windows, forecast_windows

T, G, S, FV = 24, 3, 2, 2


def row_of(x):
    """Every synthetic value is 100 * row + column, so value // 100 is the record row it came from."""
    return x // 100


class Record:
    """WeatherDataset slicing (num_past = num_future = 1) over time-coded numpy arrays."""

    def __init__(self, n):
        self.n = n
        self.state = 100.0 * np.arange(T)[:, None, None] + np.arange(S)[None, None]
        self.state = np.broadcast_to(self.state, (T, G, S)).copy()
        self.forcing = np.broadcast_to(100.0 * np.arange(T)[:, None, None] + np.arange(FV)[None, None], (T, G, FV)).copy()

    def __len__(self):
        return T - self.n - 2

    def __getitem__(self, idx):
        rows = self.state[idx:idx + 2 + self.n]
        forcing = np.stack([self.forcing[idx + 1 + k:idx + 4 + k].transpose(1, 2, 0).reshape(G, -1) for k in range(self.n)])
        return rows[:2], rows[2:], forcing, np.arange(idx + 2, idx + 2 + self.n)


def test_the_mode_has_no_default():
    for fn in (Windows, forecast_windows):
        p = inspect.signature(fn).parameters["forcing"]
        assert p.kind is p.KEYWORD_ONLY and p.default is p.empty
    with pytest.raises(TypeError):
        Windows(Record(3))


def test_oracle_hands_over_the_dataset_items_unchanged():
    base = Record(4)
    w = Windows(base, forcing=Forcing.ORACLE)
    assert len(w) == len(base)
    for idx in range(len(base)):
        old, new = base[idx], w[idx]
        for a, b in zip(old, new.batch):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(new.truth, old[1])
        assert new.boundary is new.truth
        later = w.at(idx, 0)
        for a, b in zip(new, later):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [1, 4])
def test_issue_time_persistence_gives_the_model_nothing_after_t0(n):
    base = Record(n)
    w = Windows(base, forcing=Forcing.ISSUE_TIME_PERSISTENCE)
    for issue in range(len(base)):
        t0 = issue + 1
        for idx in range(issue, len(base)):
            win = w.at(idx, issue)
            for name in ("init", "boundary", "forcing"):
                assert row_of(getattr(win, name)).max() <= t0, (name, issue, idx)
            np.testing.assert_array_equal(win.init, base.state[t0 - 1:t0 + 1])
            np.testing.assert_array_equal(win.boundary, np.broadcast_to(base.state[t0], (n, G, S)))
            np.testing.assert_array_equal(win.truth, base[idx][1])
            np.testing.assert_array_equal(win.times, base[idx][3])
            assert row_of(win.truth).min() > t0


def test_issue_time_persistence_keeps_t0_and_holds_it_for_every_later_slot():
    base = Record(3)
    w = Windows(base, forcing=Forcing.ISSUE_TIME_PERSISTENCE)
    issue = 5
    own = w[issue].forcing.reshape(3, G, FV, FORCING_WINDOW)
    np.testing.assert_array_equal(own, np.broadcast_to(base.forcing[issue + 1][None, :, :, None], own.shape))
    np.testing.assert_array_equal(own[0, :, :, 0], base[issue][2].reshape(3, G, FV, FORCING_WINDOW)[0, :, :, 0])
    later = w.at(issue + 4, issue).forcing.reshape(3, G, FV, FORCING_WINDOW)
    np.testing.assert_array_equal(later, own)
    with pytest.raises(ValueError, match="issued"):
        w.at(issue, issue + 1)
