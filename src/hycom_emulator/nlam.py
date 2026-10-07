"""Run a neural-lam command with the `hycom` datastore registered.

    python -m hycom_emulator.nlam create_graph --config_path nlam.yaml --name multiscale
    python -m hycom_emulator.nlam build_graph nlam.yaml mesh3
    python -m hycom_emulator.nlam train_model --config_path nlam.yaml --model graph_lam ...
    python -m hycom_emulator.nlam train_model ... --model crps_graph_lam --loss afcrps --members 2 --init_from <ckpt>

train_model also takes the arm flags of hycom_emulator.rea_train. With --model crps_graph_lam, --checkpoint_steps
belongs to hycom_emulator.ensemble, which takes its flags first.
"""

from __future__ import annotations

import sys

import hycom_emulator.datastore  # noqa: F401  registers DATASTORES["hycom"]
import hycom_emulator.convnet  # noqa: F401  registers the conv models and, via physics, hycom_graph_lam and hycom_wmse
import hycom_emulator.rea_loss  # noqa: F401  registers rea_wmse
from hycom_emulator import ensemble  # registers crps_graph_lam, afcrps and fcrps


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


def _exclude_source_changes() -> None:
    """With `exclude_source_changes: true` in the datastore yaml, keep only the train samples whose state rows
    (idx .. idx + 1 + ar_steps) come from one source experiment. Val and test keep every sample, so their
    scores stay comparable across arms."""
    from neural_lam.weather_dataset import WeatherDataModule
    from torch.utils.data import Subset

    from hycom_emulator.datastore import kept_windows

    original = WeatherDataModule.setup

    def setup(self, stage=None):
        original(self, stage)
        if stage not in ("fit", None) or not getattr(self._datastore, "exclude_source_changes", False):
            return
        if self._datastore.is_ensemble or self.num_past_forcing_steps > 2:
            raise ValueError("exclude_source_changes needs a single-member datastore and num_past_forcing_steps <= 2")
        train = self.train_dataset
        keep = kept_windows(train.da_state.time.values, 2 + train.ar_steps, len(train))
        print(f"exclude_source_changes: dropped {len(train) - keep.size} of {len(train)} train windows", flush=True)
        self.train_dataset = Subset(train, keep.tolist())

    WeatherDataModule.setup = setup


def build_graph(config_path: str, name: str) -> None:
    """Build graph/<name> next to the datastore config unless it is there. multiscale: neural-lam's
    create_graph (GraphCast-like, finest mesh 81x81, ~6x5 grid cells on 525x385). mesh3: the same layout
    from weather-model-graphs with a 125x125 finest mesh (~4x3 cells) and levels 25x25 and 5x5. wmg
    rounds each direction down to a power of the (odd) refinement factor: factor 3 gives 81x81 again,
    and asking for 3 cells (0.12 deg) gives 125x25, since rows are ~0.037 deg apart; 2.5 cells gives 125
    in both directions. <graph>_s<k>: <graph>'s mesh with grid
    edges for k x k cells (convnet option B)."""
    from pathlib import Path

    from neural_lam.config import load_config_and_datastore

    _, ds = load_config_and_datastore(config_path=config_path)
    out = Path(ds.root_path) / "graph" / name
    if (out / "m2g_features.pt").is_file():
        return
    base, _, stride = name.rpartition("_s")
    if base and stride.isdigit():
        from hycom_emulator.convnet import coarse_graph

        build_graph(config_path, base)
        coarse_graph(out.with_name(base), ds.get_xy("state", stacked=False), int(stride))
    elif name == "multiscale":
        from neural_lam.create_graph import cli

        cli(["--config_path", config_path, "--name", name])
    elif name == "mesh3":
        from neural_lam.create_graph_with_wmg import create_graph_from_datastore

        create_graph_from_datastore(ds, str(out), archetype="graphcast", mesh_grid_distance_ratio=2.5, level_refinement_factor=5)
    else:
        raise SystemExit(f"unknown graph {name!r}")


def main(argv: list[str]) -> None:
    command, rest = argv[1], argv[2:]
    _load_own_checkpoints()
    _fork_workers()
    _exclude_source_changes()
    if command == "create_graph":
        from neural_lam.create_graph import cli

        sys.argv = ["neural_lam.create_graph", *rest]
        cli()
    elif command == "build_graph":
        build_graph(*rest)
    elif command == "train_model":
        from neural_lam import train_model

        from hycom_emulator import rea_train

        if rea_train._value(rest, "--model", "graph_lam") == "crps_graph_lam":
            ens, rest = ensemble.split_args(rest)
            train_model.ForecasterModule = ensemble.module_factory(ens.members, ens.init_from, ens.checkpoint_steps)
        opts, rest = rea_train.split_args(rest)
        rea_train.install(opts, rest)
        train_model.main(rest)
    else:
        raise SystemExit(f"unknown command {command!r}; use create_graph, build_graph or train_model")


if __name__ == "__main__":
    main(sys.argv)
