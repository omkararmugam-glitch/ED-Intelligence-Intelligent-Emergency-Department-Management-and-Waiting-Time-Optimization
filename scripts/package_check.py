"""Clean-package check: is everything a recipient needs present, and nothing they should not get?"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    # the one-click path and everything it touches
    "Start-Demo.cmd",
    "Stop-Demo.cmd",
    "scripts/start_demo.ps1",
    "scripts/stop_demo.ps1",
    "scripts/setup.ps1",
    "scripts/run_pipeline.ps1",
    "scripts/run_api.ps1",
    "scripts/run_app.ps1",
    "scripts/run_tests.ps1",
    # docker path
    "start_docker.bat",
    "scripts/docker_up.ps1",
    "scripts/docker_down.ps1",
    "scripts/_docker_common.ps1",
    "Dockerfile",
    "docker-compose.yml",
    # research drivers + their helpers
    "scripts/robustness_study.py",
    "scripts/run_robustness_study.ps1",
    "scripts/scenario_lab.py",
    "scripts/run_scenario_lab.ps1",
    "scripts/staffing_rl.py",
    "scripts/run_staffing_rl.ps1",
    "scripts/original_generator/generate_ed_data.py",
    # docs
    "README.md",
    "SUBMISSION.md",
    "CLAUDE.md",
    "docs/real_data_readiness.md",
    "docs/demo_day_checklist.md",
    "artifacts/checkpoints/days_work.md",
    # code + config
    "api/main.py",
    "app/dashboard.py",
    "src/pipeline.py",
    "configs/default.yaml",
    "configs/column_map_mimic_ed.yaml",
    "requirements.txt",
    "pytest.ini",
    "schema.yaml",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    # the demo needs these to start at all
    "artifacts/metrics/wait_models.json",
    "data/raw/ed_v2.csv",
]

# Files the launcher scripts reference, which must therefore exist.
REFERENCED_BY_LAUNCHER = {
    "scripts/start_demo.ps1": [".venv/Scripts/python.exe", "app/dashboard.py",
                               "artifacts/metrics/wait_models.json"],
}

problems: list[str] = []
notes: list[str] = []

print("=" * 72)
print("1. REQUIRED FILES")
print("=" * 72)
missing = [p for p in REQUIRED if not (ROOT / p).exists()]
for p in REQUIRED:
    mark = "ok " if (ROOT / p).exists() else "MISSING"
    if mark != "ok ":
        problems.append(f"missing required file: {p}")
        print(f"  {mark}  {p}")
print(f"  {len(REQUIRED) - len(missing)}/{len(REQUIRED)} present")

print()
print("=" * 72)
print("2. WHAT THE LAUNCHERS REFERENCE")
print("=" * 72)
for script, targets in REFERENCED_BY_LAUNCHER.items():
    for t in targets:
        exists = (ROOT / t).exists()
        if not exists:
            problems.append(f"{script} references {t}, which does not exist")
        print(f"  {'ok ' if exists else 'MISSING'}  {t}   <- {script}")

print()
print("=" * 72)
print("3. SECRETS AND PRIVATE DATA")
print("=" * 72)
# .env may hold a key; it is gitignored, but a zipped folder would carry it.
env = ROOT / ".env"
if env.exists():
    body = env.read_text(encoding="utf-8", errors="replace")
    live = [ln for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("#") and "=" in ln
            and ln.split("=", 1)[1].strip() not in ("", '""', "''")]
    if live:
        notes.append(f".env holds {len(live)} set value(s); exclude it from any package "
                     "(it is gitignored, but a zip of the folder would include it)")
        print(f"  NOTE  .env has {len(live)} value(s) set:")
        for ln in live:
            key = ln.split("=", 1)[0].strip()
            print(f"          {key}=<redacted>")
    else:
        print("  ok    .env present but holds no set values")
else:
    print("  ok    no .env")

# Any obviously-real patient data?
suspicious = []
for p in (ROOT / "data").rglob("*.csv"):
    rel = p.relative_to(ROOT).as_posix()
    if "ed_v2" in p.name or "synthetic" in p.name or "sample" in p.name or "fixture" in p.name:
        continue
    suspicious.append(rel)
if suspicious:
    print("  CHECK these CSVs are synthetic:")
    for s in suspicious[:20]:
        print(f"          {s}")
    notes.append(f"{len(suspicious)} CSV(s) under data/ with unrecognised names")
else:
    print("  ok    no unexpected CSVs under data/")

print()
print("=" * 72)
print("4. SIZE AND BULK")
print("=" * 72)
EXCLUDE_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "runs", ".mypy_cache"}


def tree_size(base: Path) -> tuple[int, int]:
    total = count = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            try:
                total += (Path(dirpath) / f).stat().st_size
                count += 1
            except OSError:
                pass
    return total, count


total, count = tree_size(ROOT)
print(f"  package (excluding {', '.join(sorted(EXCLUDE_DIRS))}): "
      f"{total / 1e6:.1f} MB in {count} files")

big = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for f in filenames:
        p = Path(dirpath) / f
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz > 5_000_000:
            big.append((sz, p.relative_to(ROOT).as_posix()))
for sz, rel in sorted(big, reverse=True)[:12]:
    print(f"    {sz / 1e6:7.1f} MB  {rel}")
if not big:
    print("    no file over 5 MB")

print()
print("=" * 72)
print("5. STRAY FILES")
print("=" * 72)
STRAY_PATTERNS = [r"\.orig$", r"\.rej$", r"\.bak$", r"~$", r"^nul$", r"\.tmp$",
                  r"^Thumbs\.db$", r"^desktop\.ini$", r"\.log$", r"\.pyc$"]
strays = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for f in filenames:
        if any(re.search(pat, f, re.I) for pat in STRAY_PATTERNS):
            strays.append((Path(dirpath) / f).relative_to(ROOT).as_posix())
if strays:
    for s in strays[:25]:
        print(f"    {s}")
    notes.append(f"{len(strays)} stray file(s) outside runs/ and .venv/")
else:
    print("    none")

print()
print("=" * 72)
print("6. RUNTIME LEFTOVERS (must not be shipped, must be ignored)")
print("=" * 72)
gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
for path, pattern in [("runs/demo.pids", "runs/"), ("runs/logs", "runs/")]:
    ignored = pattern in gitignore
    exists = (ROOT / path).exists()
    print(f"  {path:18s} exists={exists!s:5s} covered by .gitignore '{pattern}'={ignored}")
    if exists and not ignored:
        problems.append(f"{path} exists and is not gitignored")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
if problems:
    print(f"  {len(problems)} PROBLEM(S):")
    for p in problems:
        print(f"    - {p}")
else:
    print("  no problems")
if notes:
    print(f"  {len(notes)} note(s):")
    for n in notes:
        print(f"    - {n}")
sys.exit(1 if problems else 0)
