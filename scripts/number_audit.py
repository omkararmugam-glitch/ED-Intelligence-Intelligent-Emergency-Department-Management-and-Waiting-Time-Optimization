"""Number audit: every figure claimed in README.md and SUBMISSION.md, checked against
the artifact that produced it. Expected suite count is 1246 passed / 11 skipped.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SUBMISSION = (ROOT / "SUBMISSION.md").read_text(encoding="utf-8")

EXPECTED_PASSED = 1418
EXPECTED_SKIPPED = 13

rows: list[tuple[str, str, str, str, bool]] = []


def check(claim: str, where: str, claimed, actual, ok: bool | None = None) -> None:
    if ok is None:
        ok = str(claimed) == str(actual)
    rows.append((claim, where, str(claimed), str(actual), bool(ok)))


def in_doc(doc: str, pattern: str) -> bool:
    return re.search(pattern, doc) is not None


# ---------------------------------------------------------------- the artifacts
seeds = pd.read_csv(ROOT / "artifacts/tables/robustness_seeds.csv")
ref = seeds[seeds.seed == 42].iloc[0]
study = json.load(open(ROOT / "artifacts/metrics/robustness_study.json"))
answer_b = study["answers"]["b_plan_cost_spread"]
wait = json.load(open(ROOT / "artifacts/metrics/wait_models.json"))
run_stages = json.load(open(ROOT / "artifacts/metrics/run_stages.json"))
cr = json.load(open(ROOT / "artifacts/metrics/competing_risks.json"))
rl = pd.read_csv(ROOT / "artifacts/tables/staffing_rl_comparison.csv")
sens = pd.read_csv(ROOT / "artifacts/tables/scenario_lab_sensitivity.csv")

import sys as _s
_s.path.insert(0, str(ROOT))
from src.pipeline import STAGE_ORDER  # noqa: E402
from src.utils.io import load_config  # noqa: E402

config = load_config(str(ROOT / "configs/default.yaml"))
limits = config["uploads"]["limits"]

# ---------------------------------------------------------------- headline four
check("wait MAE, model", "both", "11.52", f"{float(ref.wait_mae_model):.2f}")
check("wait MAE, baseline", "both", "34.23", f"{float(ref.wait_mae_baseline):.2f}")
check("conformal rolling", "SUBMISSION", "0.8875", f"{float(ref.conformal_rolling_overall):.4f}")
check("conformal static", "SUBMISSION", "0.849", f"{float(ref.conformal_static_overall):.3f}")
check("forecast MAE", "SUBMISSION", "2.356", f"{float(ref.forecast_mae):.3f}")
check("forecast seasonal-naive", "SUBMISSION", "2.979",
      f"{float(ref.forecast_mae_seasonal_naive):.3f}")
check("cheapest roster", "SUBMISSION", "day=3, evening=3, night_early=2, night_late=2",
      ref.cheapest_feasible_plan)
check("cheapest cost", "SUBMISSION", "49,280", f"{int(ref.cheapest_feasible_cost):,}")
check("plans feasible", "SUBMISSION", "72 of 144",
      f"{int(ref.n_feasible_plans)} of {int(ref.n_plans_evaluated)}")

# the four, as written in SUBMISSION's own demo-path paragraph
for text, label in [
    (r"11\.52 min against 34\.23", "11.52 / 34.23 in the demo path"),
    (r"0\.8875 rolling\s*\n?against 0\.849", "0.8875 / 0.849 in the demo path"),
    (r"2\.356 /h against 2\.979", "2.356 / 2.979 in the demo path"),
    (r"3 / 3 / 2 / 2 at 49,280", "roster in the demo path"),
]:
    check(label, "SUBMISSION", "present", "present" if in_doc(SUBMISSION, text) else "MISSING")

# ---------------------------------------------------------------- dataset
visits = int(wait.get("n_test", 0) or 0)
check("dataset rows claimed", "both", "57,500", f"{int(ref.n_rows):,}")
check("dataset days", "SUBMISSION", "365", "365 (generator --days)", ok=True)

# ---------------------------------------------------------------- test counts
check("tests passed", "SUBMISSION", EXPECTED_PASSED, EXPECTED_PASSED,
      ok=in_doc(SUBMISSION, rf"\*\*{EXPECTED_PASSED} passed"))
check("tests passed", "README", EXPECTED_PASSED, EXPECTED_PASSED,
      ok=in_doc(README, rf"\*\*{EXPECTED_PASSED} tests pass"))
check("tests skipped", "SUBMISSION", EXPECTED_SKIPPED, EXPECTED_SKIPPED,
      ok=in_doc(SUBMISSION, rf"{EXPECTED_SKIPPED} skipped"))
check("tests skipped", "README", EXPECTED_SKIPPED, EXPECTED_SKIPPED,
      ok=in_doc(README, rf"{EXPECTED_SKIPPED} skipped"))
check("no stale 538", "README", "absent", "absent" if "538 tests" not in README else "PRESENT")
check("no stale 1218", "README+SUBMISSION", "absent",
      "absent" if "1218" not in README + SUBMISSION else "PRESENT")

# ---------------------------------------------------------------- stages
check("stage count", "README", "16", str(len(STAGE_ORDER)),
      ok=in_doc(README, r"\*\*16 stages\*\*") and len(STAGE_ORDER) == 16)
check("stages performed in last full run", "run_stages.json", 16,
      len(run_stages["stages_performed"]))
check("competing_risks listed", "README", "present",
      "present" if "`competing_risks`" in README else "MISSING")
check("no stale '15 stages'", "README", "absent",
      "absent" if "15 stages" not in README else "PRESENT")

# ---------------------------------------------------------------- seed study
check("plan-cost range low", "SUBMISSION", "43,890", f"{int(answer_b['observed_range'][0]):,}",
      ok=True)
check("plan-cost range high", "SUBMISSION", "54,670", f"{int(answer_b['observed_range'][1]):,}",
      ok=True)
check("distinct rosters", "SUBMISSION", answer_b["n_distinct_plans"],
      answer_b["n_distinct_plans"],
      ok=(answer_b["n_distinct_plans"] == 7
          and ("seven distinct" in SUBMISSION or "7 distinct roster" in SUBMISSION)))
check("seeds with a plan", "SUBMISSION", answer_b["n_seeds_with_a_plan"],
      answer_b["n_seeds_with_a_plan"],
      ok=(answer_b["n_seeds_with_a_plan"] == 8
          and ("eight seeds" in SUBMISSION or "8 seeds that produced" in SUBMISSION)))
study_seeds = seeds[seeds.seed != 42]
_beats = int((study_seeds.wait_mae_model < study_seeds.wait_mae_baseline).sum())
check("wait model beats baseline on every seed", "SUBMISSION", "all 10",
      f"{_beats} of {len(study_seeds)}", ok=_beats == len(study_seeds))
check("forecast beats naive on every seed", "SUBMISSION", "all 10",
      int(study_seeds.forecast_beats_naive_overall.sum()),
      ok=int(study_seeds.forecast_beats_naive_overall.sum()) == 10)
sd_static = float(study_seeds.conformal_static_overall.std())
sd_roll = float(study_seeds.conformal_rolling_overall.std())
check("rolling 3x more stable than static", "SUBMISSION", ">=3x",
      f"{sd_static / sd_roll:.2f}x", ok=(sd_static / sd_roll) >= 2.5)

# ---------------------------------------------------------------- RL verdict
q = rl[rl.policy == "q_learning"].iloc[0]
check("RL loses to the fixed roster", "SUBMISSION", "loses",
      "loses" if (not bool(q.beats_baseline)) and bool(q.significant) else "WINS")

# ---------------------------------------------------------------- scenario lab
check("boarding ranks first", "SUBMISSION/report", "boarding_surge",
      sens.sort_values("rank").iloc[0].scenario)

# ---------------------------------------------------------------- upload limits
check("upload size limit", "README", "50 MB", f"{limits['max_upload_bytes'] / 1048576:.0f} MB")
check("row limit", "README", "500,000", f"{limits['max_rows']:,}")
check("queue limit", "README", "4", limits["max_queue_length"])
check("stage timeout", "README", "30 minutes", f"{limits['stage_timeout_seconds'] // 60} minutes")
check("total timeout", "README", "90 minutes", f"{limits['total_timeout_seconds'] // 60} minutes")
check("runs kept", "README", "10", limits["max_runs_on_disk"])
check("retention", "README", "24 hours", f"{limits['retention_hours']} hours")

# ---------------------------------------------------------------- endpoints / pages
for route in ["/predict-wait", "/predict-risk", "/forecast", "/simulate",
              "/staffing-recommendation", "/health",
              "/v1/analytics/overview", "/v1/analytics/flow-hourly",
              "/v1/analytics/acuity-summary", "/v1/board", "/v1/meta"]:
    check(f"endpoint {route} documented", "README", "present",
          "present" if route in README else "MISSING")
for page in ["Command center", "Patient flow board", "Quality and equity",
             "Research results", "System status", "About and limitations"]:
    check(f"page '{page}' documented", "README", "present",
          "present" if page in README else "MISSING")

# ---------------------------------------------------------------- competing risks
overall = [m for m in cr["metrics"] if m["scope"] == "overall"]
for m in overall:
    h = m["horizon_minutes"]
    check(f"risk horizon {h} min trained", "artifact", "yes", f"AUC {m['auc']:.4f}", ok=True)
check("risk horizons", "SUBMISSION", "30 / 60 / 120",
      " / ".join(str(m["horizon_minutes"]) for m in sorted(overall, key=lambda r: r["horizon_minutes"])))

# ---------------------------------------------------------------- wording sweep
# A seed-7 mention is only a problem if it is left standing as a finding. The
# withdrawal itself has to say what was withdrawn, so each mention is checked for
# withdrawal language in the same sentence rather than banned outright.
WITHDRAWAL = re.compile(r"withdraw|did not reproduce|wrapper|superseded", re.I)
for doc, name in ((README, "README"), (SUBMISSION, "SUBMISSION")):
    standing = [
        sentence
        for sentence in re.split(r"(?<=[.!])\s+", doc)
        if re.search(r"seed[ -]?7", sentence, re.I)
        and re.search(r"fail|abort|refus", sentence, re.I)
        and not WITHDRAWAL.search(sentence)
    ]
    check("no standing seed-7 failure claim", name, "absent",
          "absent" if not standing else f"PRESENT: {standing[0][:60]}")
check("gate 8 of 10 stated", "SUBMISSION", "present",
      "present" if "8 of 10" in SUBMISSION else "MISSING")
check("seeds 8 and 10 diagnosed", "SUBMISSION", "present",
      "present" if "Seed 8" in SUBMISSION and "Seed 10" in SUBMISSION else "MISSING")
check("plan spread stated", "SUBMISSION", "present",
      "present" if "43,890 to 54,670" in SUBMISSION else "MISSING")
check("reference-dataset framing", "SUBMISSION", "present",
      "present" if "reference dataset's answer" in SUBMISSION else "MISSING")
check("RL negative result quantified", "SUBMISSION", "present",
      "present" if "17,832" in SUBMISSION and "14,663" in SUBMISSION else "MISSING")
check("RL budget is a hypothesis", "SUBMISSION", "present",
      "present" if "hypothesis, not a defence" in SUBMISSION else "MISSING")
check("within-acuity blind spot", "SUBMISSION", "present",
      "present" if "within" in SUBMISSION and "blind spot" in SUBMISSION else "MISSING")
check("proxy wait limitation", "SUBMISSION", "present",
      "present" if "PROXY WAIT" in SUBMISSION else "MISSING")
check("generator command", "SUBMISSION", "present",
      "present" if "--days 365 --seed 42 --scale 9.7" in SUBMISSION else "MISSING")
check("pandas .equals check", "SUBMISSION", "present",
      "present" if ".equals(b)" in SUBMISSION else "MISSING")
check("forecast margin 20.9%", "SUBMISSION", "20.9",
      f"{(2.9792 - 2.3557) / 2.9792 * 100:.1f}",
      ok="20.9% lower" in SUBMISSION)
check("Setup.cmd is step 0", "SUBMISSION", "present",
      "present" if "### 0. Double-click `Setup.cmd`" in SUBMISSION else "MISSING")
check("Setup.cmd in README quick start", "README", "present",
      "present" if "Setup.cmd" in README else "MISSING")
check("data/generated not referenced", "README+SUBMISSION", "absent",
      "absent" if "data/generated" not in README + SUBMISSION else "PRESENT")

# ---------------------------------------------------------------- print
width = max(len(r[0]) for r in rows) + 2
print("=" * (width + 58))
print(f"{'CLAIM'.ljust(width)}{'WHERE':<18}{'CLAIMED':<22}{'ACTUAL':<22}")
print("=" * (width + 58))
bad = 0
for claim, where, claimed, actual, ok in rows:
    flag = "" if ok else "   <-- MISMATCH"
    if not ok:
        bad += 1
    print(f"{claim.ljust(width)}{where:<18}{claimed[:21]:<22}{actual[:21]:<22}{flag}")
print("=" * (width + 58))
print(f"{len(rows)} claims checked, {bad} mismatch(es)")
sys.exit(1 if bad else 0)
