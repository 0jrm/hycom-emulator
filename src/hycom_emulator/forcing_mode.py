"""The one place a neural-lam WeatherDataset is built, so every score and free run states which forcing it used.

A free forecast issued at t0 cannot know the wind, sunlight or boundary band after t0. neural-lam's WeatherDataset
hands the model all three from the truth, and every evaluator used to build one itself. Forcing names the choice:

- ORACLE: true forcing and true boundary band at every lead, after t0 included. What every score so far used.
- ISSUE_TIME_PERSISTENCE: forcing and band held at their t0 values; no value from after t0 reaches the model.

Neither is a default. forecast_windows and Windows take the mode as a required keyword; tests/test_forcing_seam.py
fails if any other module builds a WeatherDataset. Training data comes from oracle_training_data, rea_train's only.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

FORCING_WINDOW = 3  # num_past_forcing_steps = num_future_forcing_steps = 1: rows t-1, t, t+1 for the step to t


class Forcing(enum.Enum):
    ORACLE = "true daily wind and boundary band at every lead, after t0 included"
    ISSUE_TIME_PERSISTENCE = "daily wind and boundary band held at their t0 values; nothing after t0 reaches the model"


FREE_FORECAST_LABEL = {
    Forcing.ORACLE: "free forecast given the true daily wind and boundary band (no increments, no observations after t0)",
    Forcing.ISSUE_TIME_PERSISTENCE: "free forecast given the daily wind and boundary band held at t0 (no increments, no observations after t0)",
}


class Window(NamedTuple):
    """One forecast: what the model is given, and the truth it is scored on. In ORACLE boundary is truth."""

    init: object  # (2, grid, state): states at t0 - 1 and t0
    boundary: object  # (ar_steps, grid, state): the forecaster overwrites its band with these every step
    forcing: object  # (ar_steps, grid, forcing_feature * FORCING_WINDOW), feature-major, window inner
    times: object  # (ar_steps,) target times, int64 ns
    truth: object  # (ar_steps, grid, state)

    @property
    def batch(self) -> tuple:
        """neural-lam's (init_states, target_states, forcing, target_times); target_states only feeds the band."""
        return self.init, self.boundary, self.forcing, self.times


class Windows:
    """Window per WeatherDataset index. at(idx, issue) is window idx of a run issued at window issue's t0 (row issue + 1):
    a free run continued in chunks keeps holding the values of its own t0, not of the chunk's."""

    def __init__(self, base, *, forcing: Forcing):
        self.base, self.forcing = base, forcing

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Window:
        return self.at(idx, idx)

    def at(self, idx: int, issue: int) -> Window:
        init, target, forcing, times = item = self.base[idx]
        if self.forcing is Forcing.ORACLE:
            return Window(init, target, forcing, times, target)
        if not 0 <= issue <= idx:
            raise ValueError(f"window {idx} cannot belong to a run issued at window {issue}")
        issued = item if issue == idx else self.base[issue]
        return Window(*held_at_issue(issued[0], issued[2], forcing, idx - issue), times, target)


def held_at_issue(init0, forcing0, forcing, lag: int) -> tuple:
    """(init, boundary, forcing) for a window lag windows after the run's issue window, which gave init0 and forcing0.
    Forcing slots up to t0 keep their value; every later slot takes its feature's t0 value. The band is the t0 state.
    Works on numpy arrays and torch tensors alike."""
    ar, _, width = forcing.shape
    t0_slot = [c - c % FORCING_WINDOW for c in range(width)]  # step 0, window 0 is row t0
    held = forcing0[:1, :, t0_slot][[0] * ar]
    for k in range(ar):
        for w in range(FORCING_WINDOW):
            if lag + k + w <= 0:  # window w of step k is row t0 + lag + k + w
                held[k, :, w::FORCING_WINDOW] = forcing[k, :, w::FORCING_WINDOW]
    return init0, init0[[1] * ar], held


def forecast_windows(datastore, split: str, ar_steps: int, *, forcing: Forcing) -> Windows:
    from neural_lam.weather_dataset import WeatherDataset

    return Windows(WeatherDataset(datastore, split=split, ar_steps=ar_steps, num_past_forcing_steps=1,
                                  num_future_forcing_steps=1), forcing=forcing)


def oracle_training_data(datastore, split: str, ar_steps: int, past_steps: int, future_steps: int):
    """neural-lam's WeatherDataset over true forcing, as rea_train has always trained on. Issue-time training is not built."""
    from neural_lam.weather_dataset import WeatherDataset

    return WeatherDataset(datastore, split=split, ar_steps=ar_steps, num_past_forcing_steps=past_steps,
                          num_future_forcing_steps=future_steps)
