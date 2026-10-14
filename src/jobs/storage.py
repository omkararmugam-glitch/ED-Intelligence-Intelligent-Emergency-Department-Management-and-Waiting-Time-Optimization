"""Run folders: layout, isolation, listing, bundling and deletion.

One job owns one directory and writes nowhere else. That is the whole isolation
story, and it is enforced here rather than trusted: `build_run_config` rewrites
*every* path in the config to sit under the run folder, so a stage that forgets
where it is still cannot reach `artifacts/`.

The uploaded file is treated as the sensitive thing it is. It is written once, into
the run folder, and never copied, never logged, and never served back. Deleting the
job deletes the folder, and a retention sweep deletes folders nobody asked to keep.
"""

from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.jobs.settings import UploadSettings
from src.utils.io import project_root, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: The uploaded file's name inside the run folder. The user's own filename is kept
#: only as metadata: it is their text, and using it as a path is how traversal bugs
#: and encoding surprises get in.
UPLOAD_FILENAME = "upload.csv"

#: Artifact subdirectories offered for download, relative to the run's artifacts dir.
DOWNLOADABLE_DIRS = ("tables", "figures", "metrics")

#: Extensions that may be served from a run folder.
SERVABLE_SUFFIXES = frozenset({".csv", ".json", ".md", ".png", ".svg", ".txt", ".zip"})


@dataclass(frozen=True)
class RunPaths:
    """Where everything for one job lives."""

    job_id: str
    root: Path

    @property
    def upload(self) -> Path:
        """The uploaded file."""
        return self.root / UPLOAD_FILENAME

    @property
    def column_map(self) -> Path:
        """The column map built for this upload."""
        return self.root / "column_map.yaml"

    @property
    def config(self) -> Path:
        """The run's own config, with every path redirected into this folder."""
        return self.root / "config.yaml"

    @property
    def processed(self) -> Path:
        """Intermediate parquet files."""
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        """Models, metrics, figures, tables and the report."""
        return self.root / "artifacts"

    @property
    def report(self) -> Path:
        """The generated markdown report."""
        return self.artifacts / "report.md"

    @property
    def log(self) -> Path:
        """This run's pipeline log."""
        return self.root / "pipeline.log"

    @property
    def record(self) -> Path:
        """The job record, so a restarted process can still describe past jobs."""
        return self.root / "job.json"

    @property
    def bundle(self) -> Path:
        """The ZIP of tables and figures, built on demand."""
        return self.root / "results.zip"


def runs_root(settings: UploadSettings) -> Path:
    """The directory holding every run folder."""
    return resolve_path(settings.runs_dir)


def run_paths(settings: UploadSettings, job_id: str) -> RunPaths:
    """Where a given job's files live."""
    return RunPaths(job_id=job_id, root=runs_root(settings) / job_id)


def prepare_run(settings: UploadSettings, job_id: str) -> RunPaths:
    """Create an empty run folder and return its paths."""
    paths = run_paths(settings, job_id)
    for directory in (
        paths.root,
        paths.processed,
        paths.artifacts / "models",
        paths.artifacts / "metrics",
        paths.artifacts / "figures",
        paths.artifacts / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared run folder for job %s", job_id)
    return paths


def build_run_config(
    base_config: dict[str, Any], paths: RunPaths, upload_name: str | None = None
) -> dict[str, Any]:
    """Copy the config and point every path at this run's folder.

    This is the isolation boundary. `src.utils.io.resolve_path` resolves a relative
    path against the repository root, so absolute paths are written here: a stage
    cannot then land in `artifacts/` by accident, whatever its working directory.

    Args:
        base_config: The loaded default config. Not mutated.
        paths: The run's folder layout.
        upload_name: The user's own filename, kept as metadata only.

    Returns:
        A new config dict for this run.
    """
    import copy

    config = copy.deepcopy(base_config)
    config.pop("_config_path", None)

    # Outputs move through the pipeline's own redirection, so the CLI's
    # --run-name and an uploaded job cannot drift apart about where a run writes.
    from src.pipeline import redirect_to_run

    redirect_to_run(config, paths.root)

    # Inputs: the contract and the alias list are read-only and shared; the extract
    # and its column map belong to this job alone.
    config["paths"]["schema"] = str(resolve_path(base_config["paths"]["schema"]))
    config["paths"]["column_map"] = str(paths.column_map)
    config["paths"]["raw_dir"] = str(paths.root)
    config["paths"]["raw_file"] = str(paths.upload)
    config["paths"]["synthetic_dir"] = str(paths.root / "data" / "synthetic")
    config["paths"]["synthetic_file"] = str(
        paths.root / "data" / "synthetic" / "ed_visits.parquet"
    )

    # A scenario or staffing result from an unvalidated simulation is refused by the
    # job planner, never overridden from the UI. Pinned here so a config edit cannot
    # quietly turn it on for uploads.
    config.setdefault("simulation", {})["allow_invalid_calibration"] = False

    config["_run"] = {
        "job_id": paths.job_id,
        "uploaded_filename": upload_name,
    }
    return config


def write_run_config(config: dict[str, Any], paths: RunPaths) -> Path:
    """Write the run config to disk and return its path."""
    paths.config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return paths.config


def write_column_map(column_map: dict[str, Any], paths: RunPaths) -> Path:
    """Write the run's column map to disk and return its path."""
    paths.column_map.write_text(
        yaml.safe_dump(column_map, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return paths.column_map


@dataclass(frozen=True)
class ArtifactInfo:
    """One downloadable file produced by a run."""

    name: str
    kind: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form."""
        return {"name": self.name, "kind": self.kind, "size_bytes": self.size_bytes}


def list_artifacts(paths: RunPaths) -> list[ArtifactInfo]:
    """Every file a caller may download from this run, newest layout first.

    Model files are deliberately absent: a joblib is executable on load, so serving
    one over HTTP turns a file upload into code execution on whoever opens it.
    """
    found: list[ArtifactInfo] = []
    if paths.report.is_file():
        found.append(
            ArtifactInfo("report.md", "report", paths.report.stat().st_size)
        )
    for directory in DOWNLOADABLE_DIRS:
        base = paths.artifacts / directory
        if not base.is_dir():
            continue
        for item in sorted(base.rglob("*")):
            if item.is_file() and item.suffix.lower() in SERVABLE_SUFFIXES:
                found.append(
                    ArtifactInfo(
                        name=f"{directory}/{item.relative_to(base).as_posix()}",
                        kind=directory,
                        size_bytes=item.stat().st_size,
                    )
                )
    return found


def artifact_path(paths: RunPaths, name: str) -> Path:
    """Resolve a download name to a file inside the run folder.

    Args:
        paths: The run's folder layout.
        name: The artifact name exactly as `list_artifacts` reported it.

    Returns:
        The file's path.

    Raises:
        FileNotFoundError: If the name does not resolve to a servable file inside
            this run. Resolution is checked against the real path, so `..` segments,
            absolute paths and symlinks cannot escape the folder.
    """
    if name == "results.zip":
        candidate = paths.bundle
    elif name == "report.md":
        candidate = paths.report
    else:
        candidate = paths.artifacts / name

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FileNotFoundError(f"No artifact named {name!r} in this run.") from error

    root = paths.root.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"No artifact named {name!r} in this run.")
    if resolved.suffix.lower() not in SERVABLE_SUFFIXES:
        raise FileNotFoundError(f"{name!r} is not a downloadable file type.")
    return resolved


def build_bundle(paths: RunPaths) -> Path:
    """Zip this run's tables and figures, and return the archive path."""
    paths.bundle.unlink(missing_ok=True)
    with zipfile.ZipFile(paths.bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        if paths.report.is_file():
            archive.write(paths.report, "report.md")
        for directory in ("tables", "figures"):
            base = paths.artifacts / directory
            if not base.is_dir():
                continue
            for item in sorted(base.rglob("*")):
                if item.is_file() and item.suffix.lower() in SERVABLE_SUFFIXES:
                    archive.write(item, f"{directory}/{item.relative_to(base).as_posix()}")
    logger.info("Built result bundle for job %s", paths.job_id)
    return paths.bundle


def delete_run(paths: RunPaths) -> bool:
    """Delete a run folder and everything in it, including the upload.

    Returns:
        True if a folder was removed.
    """
    if not paths.root.exists():
        return False
    shutil.rmtree(paths.root, ignore_errors=True)
    # On Windows a file held open by a reader survives the first attempt; one retry
    # covers the usual case without turning deletion into a loop.
    if paths.root.exists():
        time.sleep(0.2)
        shutil.rmtree(paths.root, ignore_errors=True)
    removed = not paths.root.exists()
    logger.info("Deleted run folder for job %s (removed=%s)", paths.job_id, removed)
    return removed


def existing_run_ids(settings: UploadSettings) -> list[str]:
    """Job ids with a folder on disk, oldest first by modification time."""
    root = runs_root(settings)
    if not root.is_dir():
        return []
    folders = [item for item in root.iterdir() if item.is_dir()]
    folders.sort(key=lambda item: item.stat().st_mtime)
    return [item.name for item in folders]


def prune_runs(
    settings: UploadSettings, protected: frozenset[str] = frozenset()
) -> list[str]:
    """Delete run folders past their retention or over the on-disk limit.

    Args:
        settings: Upload settings, for the retention window and the folder cap.
        protected: Job ids that must not be deleted -- the running job and anything
            queued behind it.

    Returns:
        The job ids whose folders were deleted.
    """
    root = runs_root(settings)
    if not root.is_dir():
        return []

    deleted: list[str] = []
    cutoff = datetime.now() - timedelta(hours=settings.limits.retention_hours)

    for job_id in existing_run_ids(settings):
        if job_id in protected:
            continue
        folder = root / job_id
        modified = datetime.fromtimestamp(folder.stat().st_mtime)
        if modified < cutoff:
            if delete_run(RunPaths(job_id, folder)):
                deleted.append(job_id)

    remaining = [job for job in existing_run_ids(settings) if job not in protected]
    excess = len(remaining) - settings.limits.max_runs_on_disk
    for job_id in remaining[: max(excess, 0)]:
        if delete_run(RunPaths(job_id, root / job_id)):
            deleted.append(job_id)

    if deleted:
        logger.info("Retention removed %d run folder(s)", len(deleted))
    return deleted


def baseline_artifacts_dir(config: dict[str, Any]) -> Path:
    """The shared artifacts directory a job must never write to.

    Exported so a test can assert isolation by hashing it before and after a run.
    """
    return resolve_path(config["paths"]["artifacts_dir"])


def is_inside_project(path: Path) -> bool:
    """Whether a path sits inside the repository, for a sanity check in tests."""
    try:
        return path.resolve().is_relative_to(project_root())
    except (OSError, ValueError):
        return False
