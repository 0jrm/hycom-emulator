"""Run a neural-lam command with the `hycom` datastore registered.

    python -m hycom_emulator.nlam create_graph --config_path nlam.yaml --name multiscale
    python -m hycom_emulator.nlam train_model --config_path nlam.yaml --model graph_lam ...
"""

from __future__ import annotations

import sys

import hycom_emulator.datastore  # noqa: F401  registers DATASTORES["hycom"]


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


def main(argv: list[str]) -> None:
    command, rest = argv[1], argv[2:]
    _load_own_checkpoints()
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
