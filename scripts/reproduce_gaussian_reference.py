"""Run identified historical Gaussian recipes without changing the current model."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

REFERENCE_COMMIT = "cc73e14f25a5c9eff2da12838467de54454464ca"
# This filename belongs to the identified historical Git object, not current code.
REFERENCE_RUNNER = "scripts/run_parallel_r2.py"
SOURCE_PATHS = ("src", "scripts/generate_results.py", REFERENCE_RUNNER)
RECIPES = {
    "near_50_50": (120, 64),
    "equal_outcome": (120, 64),
    "equal_outcome_distinct_serious": (120, 32),
    "near_50_50_focused": (1200, 36),
    "equal_outcome_focused": (1200, 48),
    "equal_outcome_distinct_focused": (1200, 64),
}


def command(recipe: str, source: Path, output: Path, python: str,
            workers: int, smoke: bool = False) -> list[str]:
    """Keep the historical preset, partitioning, and common-stream arguments."""
    episodes, chunks = RECIPES[recipe]
    if workers < 1:
        raise ValueError("workers must be positive")
    grid = "equal_outcome_distinct_focused" if recipe.endswith("_serious") else recipe
    args = [python, str(source / REFERENCE_RUNNER),
            "--preset", "serious", "--sections", "regime_grid",
            "--regime-grid", grid,
            "--regime-grid-chunks", str(1 if smoke else chunks),
            "--episodes", str(1 if smoke else episodes),
            "--voi-samples", str(2 if smoke else 500),
            "--common-observations", "on",
            "--observations-per-person", "200",
            "--max-workers", str(workers), "--output-dir", str(output)]
    if smoke:
        args += ["--max-regime-grid-points", "1"]
    return args


def unpack_source(archive: bytes, destination: Path) -> None:
    """Extract regular scientific files only; never follow archive links."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        members = handle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            allowed = path.parts and (path.parts[0] == "src" or
                                       member.name in SOURCE_PATHS[1:])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe reference archive path")
            if not (member.isdir() or member.isfile()):
                raise ValueError("Reference archive contains a non-regular entry")
            if member.isfile() and not allowed:
                raise ValueError("Unexpected reference archive file")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            if member.isfile():
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = handle.extractfile(member)
                assert stream is not None
                target.write_bytes(stream.read())


def prepare_reference(repository: Path, destination: Path) -> None:
    result = subprocess.run(
        ["git", "archive", "--format=tar", REFERENCE_COMMIT, "--", *SOURCE_PATHS],
        cwd=repository, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "Historical Git object unavailable. Use a full repository clone or run "
            f"git fetch origin {REFERENCE_COMMIT}. No current-model fallback is used."
        )
    unpack_source(result.stdout, destination)


def resolve_python(value: str) -> str:
    """Resolve the caller's interpreter before entering the isolated source tree."""
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"Python executable not found: {value}")
    # Do not resolve symlinks: a virtual environment uses its invocation path.
    return os.path.abspath(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Extract the pinned source and print the command; do not simulate.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        python = resolve_python(args.python)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory; retained results are not overwritten")
    source = output / "reference_source"
    prepare_reference(args.repository, source)
    run_command = command(args.recipe, source, output / "tables", python,
                          args.workers, args.mode == "smoke")
    metadata = {"reference_commit": REFERENCE_COMMIT, "recipe": args.recipe,
                "mode": args.mode, "command": run_command,
                "scope": "Historical Gaussian model; not the current solver",
                "status": "prepared"}
    manifest = output / "reference_recipe.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    if args.prepare_only:
        return 0
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(run_command, cwd=source, env=environment, check=False)
    except OSError as error:
        metadata.update(status="failed", error=str(error))
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    metadata.update(status="completed" if result.returncode == 0 else "failed",
                    returncode=result.returncode)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
