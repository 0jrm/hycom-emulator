"""Every score and free run gets its WeatherDataset from hycom_emulator.forcing_mode, which makes the caller name a
Forcing mode. Evaluators that built their own WeatherDataset scored with true future wind and band without saying so;
this fails on the next one."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "hycom_emulator"
SEAM = "forcing_mode"
WINDOW_KEYWORDS = frozenset({"num_past_forcing_steps", "num_future_forcing_steps"})
NEEDS_MODE = frozenset({"forecast_windows", "Windows", "load_model"})
TRAINING = {"oracle_training_data": frozenset({"rea_train"})}


def _name(call: ast.Call) -> str | None:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def violations(module: str, source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name, where = _name(node), f"src/hycom_emulator/{module}.py:{node.lineno}"
        keywords = {k.arg for k in node.keywords}
        if name == "WeatherDataset":
            found.append(f"{where} builds a WeatherDataset; get it from forcing_mode.forecast_windows(..., forcing=Forcing.X)")
        elif keywords & WINDOW_KEYWORDS:
            found.append(f"{where} passes {sorted(keywords & WINDOW_KEYWORDS)}; forcing windows belong to forcing_mode")
        elif name in NEEDS_MODE and "forcing" not in keywords:
            found.append(f"{where} calls {name} without forcing=; name the mode (forcing_mode.Forcing) at the call")
        elif name in TRAINING and module not in TRAINING[name]:
            found.append(f"{where} calls {name}, which trains on true future forcing; only {sorted(TRAINING[name])} may")
    return found


def test_only_the_seam_builds_forcing_windows():
    found = [v for p in sorted(PACKAGE.glob("*.py")) if p.stem != SEAM for v in violations(p.stem, p.read_text())]
    assert not found, "\n".join(found)


def test_the_check_sees_the_shapes_the_leak_took():
    leaks = {
        "WeatherDataset(ds, split='test', ar_steps=2, num_past_forcing_steps=1, num_future_forcing_steps=1)": "builds",
        "f = lambda n: WeatherDataset(ds, 'train', n)": "builds",
        "nl.WeatherDataset(ds)": "builds",
        "make(ds, num_future_forcing_steps=1)": "passes",
        "rr.load_model(config, ckpt)": "without forcing=",
        "forecast_windows(ds, 'test', 2)": "without forcing=",
        "oracle_training_data(ds, 'train', 1, 1, 1)": "trains on true future forcing",
    }
    for source, message in leaks.items():
        found = violations("probe", source)
        assert len(found) == 1 and message in found[0], (source, found)
    assert not violations("probe", "rr.load_model(config, ckpt, forcing=Forcing.ORACLE)")
    assert not violations("rea_train", "oracle_training_data(ds, 'train', 1, 1, 1)")
