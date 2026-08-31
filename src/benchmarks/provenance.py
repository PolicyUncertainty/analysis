"""Git provenance helpers for the dcegm engine benchmark."""

import subprocess
from pathlib import Path

DCEGM_SUBMODULE_DIR = Path(__file__).resolve().parents[2] / "submodules" / "dcegm"


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(DCEGM_SUBMODULE_DIR), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def get_dcegm_git_info():
    """Branch, short commit hash, and dirty-flag of the checked-out dcegm submodule."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"branch": branch, "commit": commit, "dirty": dirty}
