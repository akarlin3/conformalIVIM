"""Tests for PEP 561 typing marker and CLI smoke-test entry point."""
from __future__ import annotations

import time
from pathlib import Path

import conformal_ivim
from conformal_ivim.cli import main


def test_pep561_py_typed_marker_present():
    pkg_dir = Path(conformal_ivim.__file__).resolve().parent
    marker = pkg_dir / "py.typed"
    assert marker.is_file(), "PEP 561 py.typed marker must exist inside conformal_ivim"


def test_cli_version_and_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "conformal-ivim" in out


def test_cli_demo_runs_fast_and_verifies_methods(capsys):
    t0 = time.perf_counter()
    rc = main(["demo"])
    elapsed = time.perf_counter() - t0
    out = capsys.readouterr().out

    assert rc == 0
    assert elapsed < 5.0, f"CLI demo took {elapsed:.2f}s (expected < 5.0s)"
    for expected in (
        "Split-Conformal",
        "CQR",
        "Mondrian-Split",
        "Mondrian-CQR",
        "In-Distribution Test Batch : fires=False",
        "Severe SNR/Domain Shift    : fires=True",
    ):
        assert expected in out
