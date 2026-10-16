"""Shared setup: make pipeline/ and train/ importable, and give every test a fake AWS."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "pipeline", ROOT / "train", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from fakes import World  # noqa: E402


@pytest.fixture
def world(capsys):
    return World(capsys)
