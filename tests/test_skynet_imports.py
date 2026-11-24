"""Skynet's venv has no gom_da or netCDF4, and modules get run there by hand as well as from scripts/skynet.
So the RCC-only side of the package is declared here: only those modules may load gom_da or netCDF4 at import time,
directly or through another hycom_emulator module."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "hycom_emulator"
RCC_ONLY_PACKAGES = frozenset({"gom_da", "netCDF4"})
RCC_ONLY_MODULES = frozenset({"background", "build_store", "catalog", "forcing", "system", "verify_store"})


def load_time_imports(module: str) -> list[tuple[str, int]]:
    return load_time_imports_of(ast.parse((PACKAGE / f"{module}.py").read_text()))


def load_time_imports_of(tree: ast.Module) -> list[tuple[str, int]]:
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, node.lineno))
    return found


def rcc_bound() -> dict[str, str]:
    """Each module that cannot import without gom_da or netCDF4, with the import that binds it."""
    modules = {p.stem: load_time_imports(p.stem) for p in PACKAGE.glob("*.py")}
    bound: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for module, imports in modules.items():
            if module in bound:
                continue
            for name, line in imports:
                top, _, rest = name.partition(".")
                child = rest.split(".")[0]
                if top in RCC_ONLY_PACKAGES:
                    bound[module] = f"src/hycom_emulator/{module}.py:{line} imports {name}"
                elif top == "hycom_emulator" and child in bound:
                    bound[module] = f"src/hycom_emulator/{module}.py:{line} imports {name} ({bound[child]})"
                else:
                    continue
                changed = True
                break
    return bound


def skynet_entry_modules() -> set[str]:
    text = "\n".join(p.read_text() for p in (ROOT / "scripts" / "skynet").iterdir() if p.is_file())
    return set(re.findall(r"hycom_emulator[./]([a-z_0-9]+)", text))


def test_only_declared_modules_need_rcc_packages_at_import():
    bound = rcc_bound()
    stray = sorted(set(bound) - RCC_ONLY_MODULES)
    assert not stray, "\n".join(
        f"{bound[m]}, so {m} cannot run on skynet. Import gom_da/netCDF4 inside the function that needs it, "
        f"or add {m} to RCC_ONLY_MODULES if it never runs there."
        for m in stray
    )


def test_tests_of_rcc_only_modules_skip_without_gom_da():
    """CI has no gom_da: a test file that loads an RCC-only module must skip before it, not fail collection."""
    bound = set(rcc_bound()) | {"gom_da"}
    unguarded = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        text = path.read_text()
        for name, line in load_time_imports_of(ast.parse(text)):
            top, _, rest = name.partition(".")
            if top == "gom_da" or (top == "hycom_emulator" and rest.split(".")[0] in bound):
                guard = text.find('pytest.importorskip("gom_da"')
                if guard == -1 or text[:guard].count("\n") + 1 > line:
                    unguarded.append(
                        f"tests/{path.name}:{line} loads {name}, which needs gom_da; put "
                        f'pytest.importorskip("gom_da", reason=...) above it so CI skips the file.'
                    )
                break
    assert not unguarded, "\n".join(unguarded)


def test_skynet_scripts_run_no_rcc_only_module():
    assert not skynet_entry_modules() & RCC_ONLY_MODULES
