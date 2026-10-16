"""Shared setup: make pipeline/ and train/ importable, and give every test a fake AWS."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "pipeline", ROOT / "train", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from fakes import World  # noqa: E402

# Folders that tools generate (or that hold git's own bookkeeping). A scan of "what is committed" must never read them:
# on a CI runner .git/ holds fetched refs and shas; .venv/ holds other people's code.
GENERATED_DIRS = {".git", ".terraform", ".venv", "venv", "__pycache__", ".pytest_cache", "output", "model"}


def repo_files(root: Path = ROOT) -> list[Path]:
    """The files of the repository that git tracks (or would track): `git ls-files` when `root` is a git checkout and git
    is installed, otherwise a walk of the folder that does not enter GENERATED_DIRS or another
    checkout nested inside it (CI may check a sibling repository out into a subfolder). Either way nothing from a generated
    folder, and never a .pyc."""
    names: list[Path] | None = None
    if (root / ".git").exists() and shutil.which("git"):
        done = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                              capture_output=True, text=True)
        if done.returncode == 0:
            names = [root / n for n in done.stdout.split("\0") if n]
    if names is None:
        names = []
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in GENERATED_DIRS and not (Path(folder) / d / ".git").exists()]            # prune: do not even descend
            names.extend(Path(folder) / f for f in files)
    return sorted(p for p in names if p.is_file() and p.suffix != ".pyc"
                  and not GENERATED_DIRS.intersection(p.relative_to(root).parts))


@pytest.fixture
def world(capsys):
    return World(capsys)
