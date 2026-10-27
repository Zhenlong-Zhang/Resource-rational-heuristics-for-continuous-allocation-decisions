"""Check precision-entry refusals without scans, simulations or model fits."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/plan_monte_carlo_precision.py"


@pytest.fixture
def precision():
    definition = importlib.util.spec_from_file_location("precision_entry_under_test", ENTRY)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def test_help_does_not_require_output_or_scheduler(tmp_path):
    result = subprocess.run([sys.executable, str(ENTRY), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert "--output" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_runtime_change_is_rejected_before_output(precision, tmp_path, monkeypatch):
    output = tmp_path / "changed_runtime"
    monkeypatch.setattr(precision.platform, "python_version", lambda: "0.0.0")
    monkeypatch.setattr(sys, "argv", [str(ENTRY), "--output", str(output)])
    with pytest.raises(RuntimeError, match="Changed precision runtime"):
        precision.main()
    assert not output.exists()


def test_existing_output_is_not_overwritten(precision, tmp_path, monkeypatch):
    # Matching labels reach only mkdir; this fixture never calculates a result.
    monkeypatch.setattr(precision.platform, "python_version", lambda: "3.10.9")
    monkeypatch.setattr(precision.np, "__version__", "1.23.5")
    monkeypatch.setattr(precision.scipy, "__version__", "1.10.0")
    marker = tmp_path / "keep.txt"
    marker.write_text("existing output")
    monkeypatch.setattr(sys, "argv", [str(ENTRY), "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        precision.main()
    assert marker.read_text() == "existing output"


def test_missing_reader_does_not_produce_completed_result(precision, tmp_path, monkeypatch):
    monkeypatch.setattr(precision.platform, "python_version", lambda: "3.10.9")
    monkeypatch.setattr(precision.np, "__version__", "1.23.5")
    monkeypatch.setattr(precision.scipy, "__version__", "1.10.0")
    orphan = tmp_path / "orphan.py"
    orphan.write_text("# source fixture; no paired R reader\n")
    monkeypatch.setattr(precision, "__file__", str(orphan))
    output = tmp_path / "incomplete"
    monkeypatch.setattr(sys, "argv", [str(orphan), "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        precision.main()
    assert not (output / "result.json").exists()
