"""Run a neural-lam command with the `hycom` datastore registered.

    python -m hycom_emulator.nlam create_graph --config_path nlam.yaml --name multiscale
    python -m hycom_emulator.nlam train_model --config_path nlam.yaml --model graph_lam ...
"""

from __future__ import annotations

import sys

import hycom_emulator.datastore  # noqa: F401  registers DATASTORES["hycom"]


def main(argv: list[str]) -> None:
    command, rest = argv[1], argv[2:]
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
