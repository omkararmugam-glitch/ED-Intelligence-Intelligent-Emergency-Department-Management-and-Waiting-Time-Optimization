"""Child-process entry point: run one job's folder to completion.

    python -m src.jobs.worker runs/<job_id>

Running the work in a separate process rather than a thread buys three things that
matter for a service a stranger can upload to:

* **Cancellation actually cancels.** A Python thread cannot be interrupted while it
  is inside LightGBM's fit or a SimPy loop; a process can be terminated.
* **Timeouts are enforceable** for the same reason.
* **A crash is contained.** A segfault in a native library kills this process, not
  the API serving everyone else.

The cost is that progress cannot be returned in memory, which is why
`src.jobs.runner.Progress` publishes a JSON file the supervisor polls.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.jobs.models import JobStatus
from src.jobs.preconditions import FULL_ORDER
from src.jobs.runner import JobRunner, Progress, load_spec
from src.jobs.storage import RunPaths
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the job whose folder is named on the command line.

    Returns:
        0 when the job finished or was cancelled cleanly, 1 when it failed.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write("usage: python -m src.jobs.worker <run_dir>\n")
        return 2

    root = Path(arguments[0]).resolve()
    if not root.is_dir():
        sys.stderr.write(f"run folder not found: {root}\n")
        return 2

    paths = RunPaths(job_id=root.name, root=root)
    progress_path = root / "progress.json"

    try:
        config, options, roster = load_spec(paths)
    except Exception as error:  # noqa: BLE001 - nothing upstream can report this
        progress = Progress.start(progress_path, list(FULL_ORDER))
        progress.finish(
            JobStatus.FAILED,
            f"The job could not be started: {type(error).__name__}: {error}",
        )
        return 1

    setup_logging(config["logging"]["level"], config["logging"].get("file"))
    logger.info("Worker starting for job %s", paths.job_id)

    progress = Progress.start(progress_path, list(FULL_ORDER))
    runner = JobRunner(
        paths=paths,
        config=config,
        options=options,
        roster=roster,
        progress=progress,
    )
    runner.run()

    status = progress.state.get("status")
    logger.info("Worker finished for job %s with status %s", paths.job_id, status)
    return 0 if status in (JobStatus.SUCCEEDED.value, JobStatus.CANCELLED.value) else 1


if __name__ == "__main__":
    sys.exit(main())
