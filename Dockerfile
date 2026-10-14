# ED Intelligence images: one file, two targets.
#
#   docker build --target api .   -> the FastAPI service and the upload job engine
#   docker build --target app .   -> the Streamlit dashboard
#
# `docker-compose.yml` builds both from here. One file rather than two because the
# base image, its digest pin and the non-root convention are the same for both, and
# a pin that lives in two files drifts.
#
# Two things shape these images:
#
# 1. **Artifacts are never baked in.** They are mounted read-only at runtime, so a
#    retrain does not mean a rebuild, and no model, extract or patient-shaped file
#    is ever carried inside an image.
# 2. **An upload job runs the whole pipeline inside the api container.**
#    `src.jobs.worker` is launched there as a subprocess, so the api image needs the
#    training stack (requirements-api.txt), not just the serving subset. The
#    dashboard loads no model at all and gets requirements-app.txt, which is why it
#    is a fraction of the size.
#
# Dependencies are installed in builder stages and copied in, which keeps pip
# metadata and caches out of the final images. Nothing is compiled: every pin ships
# a cp312 manylinux wheel, so no stage installs build-essential.
#
# The base image is pinned by tag *and* digest, so a rebuild tomorrow is the same
# build. Refresh both together:
#   docker pull python:3.12-slim
#   docker image inspect python:3.12-slim --format "{{index .RepoDigests 0}}"

ARG PYTHON_BASE=python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

# ============================================================== shared settings
FROM ${PYTHON_BASE} AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================================== api: dependencies
FROM base AS api-deps
WORKDIR /build
# Dependencies first and alone, so editing code does not reinstall them.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ============================================================== api: runtime
FROM base AS api

# libgomp1 is the OpenMP runtime LightGBM links against: `import lightgbm` fails
# without it on slim. It is the only OS package added, and without recommends.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# matplotlib writes a font cache on first use and $HOME may be read-only, so point
# it at the tmpfs. ED_API_HOST=0.0.0.0 is correct *inside* a container: the port is
# published to the host loopback only, and compose decides that, not this file.
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV ED_API_HOST=0.0.0.0
ENV ED_API_PORT=8000

COPY --from=api-deps /install /usr/local

WORKDIR /app

# Code only. No data, no artifacts, no tests, no notebooks -- see .dockerignore.
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY schema.yaml ./

# A fixed uid, so a bind-mounted runs/ folder has predictable ownership. The
# directories are created here because they are mount points and the runtime
# filesystem is read-only.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
 && mkdir -p /app/runs /app/artifacts /app/data/processed \
 && chown -R appuser:appuser /app
USER 10001

EXPOSE 8000

# The service's own readiness check: an unhealthy container is one that cannot
# answer, not merely one whose port is open. urllib, so the image needs no curl.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD ["python", "-c", "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('ED_API_PORT','8000') + '/health', timeout=4).status == 200 else 1)"]

# `exec` so uvicorn becomes the direct child of the init process and SIGTERM
# reaches it: `docker compose stop` must stop the service, not time it out.
CMD ["sh", "-c", "exec uvicorn api.main:app --host \"${ED_API_HOST:-0.0.0.0}\" --port \"${ED_API_PORT:-8000}\""]

# ============================================================== app: dependencies
FROM base AS app-deps
WORKDIR /build
COPY requirements-app.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-app.txt

# ============================================================== app: runtime
FROM base AS app

# No OS packages: nothing in the dashboard needs OpenMP or a compiler. Streamlit
# is configured through environment variables rather than a config file, so
# compose and `docker inspect` can both show what it was given.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV ED_APP_HOST=0.0.0.0
ENV ED_APP_PORT=8501

COPY --from=app-deps /install /usr/local

WORKDIR /app

# The dashboard and the theme file it reads. No src/, no api/, no tests.
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

# A fixed uid, distinct from the API user.
RUN useradd --create-home --uid 10002 --shell /usr/sbin/nologin appuser \
 && mkdir -p /app/artifacts /app/data/processed \
 && chown -R appuser:appuser /app
USER 10002

EXPOSE 8501

# Streamlit's own readiness endpoint, reached with urllib so no curl is needed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('ED_APP_PORT','8501') + '/_stcore/health', timeout=4).status == 200 else 1)"]

# `exec` for the same reason as the api target: signals must reach Streamlit.
CMD ["sh", "-c", "exec streamlit run app/dashboard.py --server.address=\"${ED_APP_HOST:-0.0.0.0}\" --server.port=\"${ED_APP_PORT:-8501}\" --server.headless=true --browser.gatherUsageStats=false"]
