"""mem_guard must stop the trainer itself, not only the timeout(1) wrapper train_rea.sh runs it under."""

import subprocess
from pathlib import Path

MEM_GUARD = Path(__file__).parents[1] / "scripts/skynet/mem_guard.sh"
MARKER = "987.654"  # a sleep length no other process uses


def survivors():
    out = subprocess.run(["pgrep", "-f", f"sleep {MARKER}"], capture_output=True, text=True)
    return out.stdout.split()


def test_stop_kills_the_child_that_timeout_moved_to_its_own_group(tmp_path):
    log = tmp_path / "guard.log"
    # HOST_GB=-1 trips the host-memory stop on the first check. As in train_rea.sh, a shell runs timeout as a child,
    # which moves itself and sleep to a process group of its own (the trailing true keeps bash from exec'ing it)
    code = subprocess.run(
        ["bash", str(MEM_GUARD), str(log), "bash", "-c", f"timeout 600 sleep {MARKER}; true"],
        env={"PATH": "/usr/bin:/bin", "HOST_GB": "-1", "GRACE_S": "1"},
        timeout=60,
    ).returncode
    try:
        assert code == 1
        assert "mem_guard killed" in log.read_text()
        assert survivors() == []
    finally:
        subprocess.run(["pkill", "-f", f"sleep {MARKER}"])
