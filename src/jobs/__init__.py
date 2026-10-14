"""The upload workflow: run the whole pipeline on a file a user just handed us.

A job is one uploaded CSV taken from bytes on the wire to a report, inside a run
folder of its own. Three properties hold throughout, and the modules here exist to
keep them true:

* **Isolation.** A job reads and writes only `runs/<job_id>/`. The baseline
  `artifacts/` that the serving endpoints load at startup are never opened for
  writing, so an upload can never change what `/predict-wait` answers.
* **Nothing silent.** A required column is never guessed. A stage that cannot run
  says so in plain English instead of failing the job or, worse, producing a number
  from data that does not support it.
* **Nothing kept.** The upload lives in its run folder and nowhere else, never in a
  log, and the folder is deleted on request and automatically after a retention
  window.

Modules:

    settings       limits, requirements and label maps, read from the config
    mapping        detect the file's dialect and columns; build the run's column map
    roster         the doctor roster the simulation needs, validated
    preconditions  which stages this data can support, and why not for the rest
    runner         execute pipeline stages inside one run folder
    engine         the queue, the worker thread, retention and the job registry
    storage        run folder layout, artifact listing, bundling and deletion
"""
