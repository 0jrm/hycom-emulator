"""Run a neural-lam command with the `hycom` datastore registered.

    python -m hycom_emulator.nlam create_graph --config_path nlam.yaml --name multiscale
    python -m hycom_emulator.nlam train_model --config_path nlam.yaml --model graph_lam ...
"""

from __future__ import annotations

import sys

import hycom_emulator.datastore  # noqa: F401  registers DATASTORES["hycom"]
import hycom_emulator.physics  # noqa: F401  registers hycom_graph_lam and hycom_wmse


def _load_own_checkpoints() -> None:
    """neural-lam resumes with Trainer.fit(ckpt_path=...) without weights_only. torch >= 2.6 then
    defaults to weights_only=True, which rejects neural-lam's own checkpoints (they pickle the run's
    argparse Namespace). These commands only load checkpoints our runs wrote, so load them fully."""
    import pytorch_lightning as pl

    for name in ("fit", "validate", "test", "predict"):
        original = getattr(pl.Trainer, name)

        def call(self, *args, _original=original, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original(self, *args, **kwargs)

        setattr(pl.Trainer, name, call)


def _fork_workers() -> None:
    """neural-lam starts DataLoader workers with spawn because fork hangs with dask. Spawn pickles
    the dataset into every worker; with a memory-mapped pack that is a full copy per worker (one
    run reached 194 GB in the trainer and 93 GB per worker). Our training data has no dask, so
    fork them and let the workers share the parent's pages."""
    from neural_lam.weather_dataset import WeatherDataModule

    original = WeatherDataModule.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.multiprocessing_context is not None:
            self.multiprocessing_context = "fork"

    WeatherDataModule.__init__ = init


def main(argv: list[str]) -> None:
    command, rest = argv[1], argv[2:]
    _load_own_checkpoints()
    _fork_workers()
    if command == "create_graph":
        from neural_lam.create_graph import cli

        sys.argv = ["neural_lam.create_graph", *rest]
        cli()
    elif command == "train_model":
        from neural_lam.train_model import main as train

        train(rest)
    else:
        raise SystemExit(f"unknown command {command!r}; use create_graph or train_model")


if __name__ == "__main__":
    main(sys.argv)
