"""Can a prolonged wait be predicted for a *high-acuity* patient at all?

**The failure this module investigates.** The shipped prolonged-wait classifier uses a
pooled target -- "wait > 30 minutes" -- and on the baseline run its false-negative rate
for ESI 2 is 1.000 and for ESI 1 is undefined. ESI 2 contributes 29 positive cases out
of 1,171 patients on the test block and ESI 1 contributes 1 out of 188. A pooled
objective buys its recall from ESI 3-5, where almost all the positives are, and the
high-acuity patients it abandons are exactly the ones for whom a long wait matters
most.

`src/models/triage_thresholds.py` already addresses the *decision rule*: it fits a
threshold per acuity where there are enough positives to fit one honestly, and falls
back to the pooled threshold where there are not. ESI 1 and 2 fall back, every time.
That module can only move an existing score around; it cannot create positives.

**This module asks whether anything else helps**, and compares four approaches on the
same split, the same queue features and the same bootstrap procedure:

``pooled``
    The shipped classifier and its pooled threshold. The control.
``acuity_weighted``
    Same target and features, but LightGBM sample weights that up-weight high-acuity
    rows, so the fitted model pays more for missing them.
``high_acuity_only``
    A separate classifier fitted on ESI 1-2 rows alone, so its loss is not diluted by
    the 95% of the department that is ESI 3-5.
``acuity_target``
    A different *target*: "exceeded the acuity-specific target wait" rather than
    "waited more than 30 minutes". An ESI 2 patient who waits 20 minutes has breached a
    15-minute target but not a 30-minute one, so this creates positives where the
    pooled label had none -- not by manufacturing signal, but by asking a question
    whose answer varies within the high-acuity group.

**What this module will not do.** It reports positive counts and confidence intervals
on every row, and where a cell rests on too few positives to estimate anything it says
so rather than printing a number. A recall of 1.000 over one positive case is not a
result, and nothing here will present it as one. The per-acuity targets it uses are
ASSUMED illustrative values from the config, not clinical standards; see
`features.acuity_target_wait_minutes` in `configs/default.yaml`.

Decision support only: nothing here triages a patient or alters care.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np
import pandas as pd

from src.data.schema import Schema
from src.features.build import make_design_matrix
from src.models.triage_thresholds import (
    TriageThresholds,
    bootstrap_metric_ci,
    choose_per_triage_thresholds,
)
from src.models.wait_time import ProlongedWaitClassifier, choose_decision_threshold
from src.utils.logging import get_logger
from src.utils.seed import derive_seed

logger = get_logger(__name__)

#: Acuity levels this module treats as "high acuity".
HIGH_ACUITY_LEVELS = (1, 2)

#: Below this many positives on the reporting block, a per-acuity recall/FNR is not
#: reported as a number. Matches `fairness.min_positives_for_fnr`'s intent: a rate
#: estimated from a handful of cases has a confidence interval wider than the unit
#: interval it lives in.
MIN_POSITIVES_TO_REPORT = 20

#: The name each approach is reported under, in the order they are compared.
APPROACH_ORDER = ("pooled", "acuity_weighted", "high_acuity_only", "acuity_target")

#: Carried into every artifact, because the targets are not a clinical standard.
TARGET_CAVEAT = (
    "The per-acuity target waits are ASSUMED ILLUSTRATIVE VALUES set in "
    "features.acuity_target_wait_minutes, chosen to be plausible and editable. They "
    "are NOT clinical standards and carry no endorsement. Changing them changes which "
    "patients count as positive and therefore every number in this table."
)


def acuity_target_minutes(config: dict[str, Any]) -> dict[int, float]:
    """Per-acuity target wait in minutes, from the config.

    Args:
        config: Full pipeline config; reads `features.acuity_target_wait_minutes`.

    Returns:
        Mapping of ESI level to target minutes.

    Raises:
        KeyError: If the config block is absent, rather than silently inventing targets.
    """
    block = config["features"].get("acuity_target_wait_minutes")
    if not block:
        raise KeyError(
            "features.acuity_target_wait_minutes is missing from the config. This "
            "module will not invent target waits; add the block (see "
            "configs/default.yaml) or do not call it."
        )
    return {int(level): float(minutes) for level, minutes in block.items()}


def add_acuity_target_label(
    frame: pd.DataFrame,
    config: dict[str, Any],
    target_column: str = "wait_minutes",
    out_column: str = "exceeds_acuity_target",
) -> pd.DataFrame:
    """Label each row against its own acuity's target wait.

    Like `prolonged_wait`, this is an outcome and must never reach a design matrix.
    It is derived from `wait_minutes`, which is already on the leakage blocklist, and
    the design matrices in this module are built by name from the feature sets, so the
    column cannot be selected into one.

    Args:
        frame: Table with `target_column` and `triage_level`.
        config: Full pipeline config.
        target_column: Column holding the realised wait.
        out_column: Name of the label column to add.

    Returns:
        A copy with `out_column` added (nullable Int8; null where the wait or the
        acuity is unknown).

    Raises:
        KeyError: If `target_column` or `triage_level` is absent.
    """
    for column in (target_column, "triage_level"):
        if column not in frame.columns:
            raise KeyError(f"Cannot label against acuity targets without {column!r}")

    targets = acuity_target_minutes(config)
    out = frame.copy()
    wait = pd.to_numeric(out[target_column], errors="coerce")
    level = pd.to_numeric(out["triage_level"], errors="coerce")
    threshold = level.map(targets)

    known = wait.notna() & threshold.notna()
    out[out_column] = (wait > threshold).astype("Int8").where(known)

    unmapped = sorted(set(level.dropna().unique()) - set(targets))
    if unmapped:
        # Not an error: the validator already restricts triage to 1-5, so this only
        # fires if the config omits a level that is present in the data.
        logger.warning(
            "No acuity target configured for ESI level(s) %s; %d row(s) left unlabelled",
            unmapped,
            int((~known & wait.notna()).sum()),
        )

    for esi, minutes in sorted(targets.items()):
        mask = level == esi
        if not mask.any():
            continue
        positives = int(out.loc[mask & known, out_column].sum())
        logger.info(
            "Acuity target ESI %d (> %.0f min): %d of %d labelled rows positive (%.1f%%)",
            esi,
            minutes,
            positives,
            int((mask & known).sum()),
            100.0 * positives / max(int((mask & known).sum()), 1),
        )
    return out


def acuity_sample_weights(
    levels: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-row training weights that up-weight high-acuity patients.

    The weights come from `wait_model.classifier.acuity_sample_weights` when it is
    present. Otherwise they are derived so that each acuity level contributes the same
    total weight as the most common level -- inverse-frequency weighting, which needs
    no hand-tuned constant and adapts to a dataset with a different acuity mix.

    Args:
        levels: ESI level per training row.
        config: Full pipeline config.

    Returns:
        Tuple of (weight per row, the weight used for each level).
    """
    configured = config["wait_model"]["classifier"].get("acuity_sample_weights")
    level_array = np.asarray(levels)

    if configured:
        weights = {int(k): float(v) for k, v in configured.items()}
        source = "wait_model.classifier.acuity_sample_weights"
    else:
        counts = pd.Series(level_array).value_counts()
        largest = float(counts.max())
        weights = {int(level): largest / float(count) for level, count in counts.items()}
        source = "inverse frequency (no acuity_sample_weights in config)"

    per_row = np.array([weights.get(int(level), 1.0) for level in level_array], dtype="float64")
    logger.info(
        "Acuity sample weights from %s: %s",
        source,
        {level: round(weight, 3) for level, weight in sorted(weights.items())},
    )
    return per_row, {str(level): round(weight, 4) for level, weight in sorted(weights.items())}


@dataclass
class ApproachResult:
    """Everything one approach produced, before it is flattened into a table."""

    name: str
    description: str
    label_definition: str
    #: Rows the approach can score at all: ESI 1-2 only, for `high_acuity_only`.
    scope: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _score_rows(
    truth: np.ndarray,
    predicted: np.ndarray,
    n_resamples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Recall, precision and FNR with bootstrap intervals, or nulls when unreportable.

    Args:
        truth: Binary labels.
        predicted: Binary hard predictions.
        n_resamples: Bootstrap resamples.
        seed: Global seed, for a derived stream.
        label: Identifier mixed into the bootstrap's seed.

    Returns:
        A dict of metric values and CI bounds. Where there are fewer than
        `MIN_POSITIVES_TO_REPORT` positives, the rate fields are None and
        `reportable` is False -- the counts are still given, because the counts are
        the finding.
    """
    n_positive = int(truth.sum())
    true_positive = int(((truth > 0) & (predicted > 0)).sum())
    false_negative = int(((truth > 0) & (predicted == 0)).sum())
    false_positive = int(((truth == 0) & (predicted > 0)).sum())

    row: dict[str, Any] = {
        "n": int(truth.size),
        "n_positive": n_positive,
        "tp": true_positive,
        "fn": false_negative,
        "fp": false_positive,
        "reportable": n_positive >= MIN_POSITIVES_TO_REPORT,
    }

    if n_positive == 0:
        row.update(
            recall=None, fnr=None, precision=None,
            unreportable_reason="no positive cases on this block: nothing to recall",
        )
        for metric in ("recall", "fnr", "precision"):
            row[f"{metric}_ci_low"] = None
            row[f"{metric}_ci_high"] = None
        return row

    recall = true_positive / n_positive
    fnr = false_negative / n_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else None
    )
    row.update(
        recall=round(recall, 4),
        fnr=round(fnr, 4),
        precision=round(precision, 4) if precision is not None else None,
    )
    for metric in ("recall", "fnr", "precision"):
        low, high = bootstrap_metric_ci(
            truth.astype("float64"), predicted, metric, seed, label, n_resamples
        )
        row[f"{metric}_ci_low"] = round(low, 4) if np.isfinite(low) else None
        row[f"{metric}_ci_high"] = round(high, 4) if np.isfinite(high) else None

    if not row["reportable"]:
        row["unreportable_reason"] = (
            f"{n_positive} positive case(s) < {MIN_POSITIVES_TO_REPORT}: the interval "
            "is too wide for the point estimate to mean anything"
        )
    else:
        row["unreportable_reason"] = None
    return row


def _evaluate_scores(
    result: ApproachResult,
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    val_levels: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_levels: np.ndarray,
    config: dict[str, Any],
) -> ApproachResult:
    """Choose thresholds on validation, then score the test block per acuity.

    The threshold -- pooled and, where there are enough positives, per acuity -- is
    chosen on the validation block and then frozen. The test block is scored once.

    Args:
        result: The approach being filled in.
        val_labels: Validation labels.
        val_scores: Validation scores.
        val_levels: Validation acuity levels.
        test_labels: Test labels.
        test_scores: Test scores.
        test_levels: Test acuity levels.
        config: Full pipeline config.

    Returns:
        `result`, with `rows` and `thresholds` populated.
    """
    seed = int(config["seed"])
    n_resamples = int(config.get("fairness", {}).get("bootstrap_resamples", 400))

    thresholds: TriageThresholds = choose_per_triage_thresholds(
        labels=val_labels,
        scores=val_scores,
        levels=val_levels,
        config=config,
        source_block="val",
    )
    result.thresholds = thresholds.to_dict()

    def add(scope: str, mask: np.ndarray, level: Hashable | None) -> None:
        """Score one slice of the test block under this approach's frozen rule."""
        if not mask.any():
            return
        subset_truth = test_labels[mask].astype("float64")
        subset_predicted = thresholds.apply(test_scores[mask], test_levels[mask])
        row = _score_rows(
            subset_truth, subset_predicted, n_resamples, seed, f"{result.name}:{scope}"
        )
        row.update(
            approach=result.name,
            scope=scope,
            threshold=(
                round(thresholds.for_level(level), 4) if level is not None else None
            ),
            threshold_source=(
                thresholds.reasons.get(level, "pooled") if level is not None else "mixed"
            ),
            val_positives=(
                int(thresholds.positives.get(level, 0)) if level is not None else None
            ),
        )
        result.rows.append(row)

    add("overall", np.ones(test_labels.size, dtype=bool), None)
    for level in sorted(pd.unique(test_levels), key=str):
        add(f"ESI {level}", test_levels == level, level)

    high_mask = np.isin(test_levels.astype("float64"), HIGH_ACUITY_LEVELS)
    if high_mask.any():
        subset_truth = test_labels[high_mask].astype("float64")
        subset_predicted = thresholds.apply(test_scores[high_mask], test_levels[high_mask])
        row = _score_rows(
            subset_truth, subset_predicted, n_resamples, seed, f"{result.name}:high"
        )
        row.update(
            approach=result.name,
            scope="ESI 1-2 pooled",
            threshold=None,
            threshold_source="mixed",
            val_positives=int(
                sum(
                    thresholds.positives.get(level, 0)
                    for level in thresholds.positives
                    if float(level) in HIGH_ACUITY_LEVELS
                )
            ),
        )
        result.rows.append(row)

    return result


def _train_classifier(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    val_features: pd.DataFrame,
    val_labels: np.ndarray,
    config: dict[str, Any],
    feature_set: str,
    tag: str,
    sample_weight: np.ndarray | None = None,
) -> ProlongedWaitClassifier:
    """Fit a LightGBM classifier, optionally weighted, with its threshold frozen on val.

    This mirrors `src.models.wait_time.train_prolonged_classifier` and exists only
    because that function takes no `sample_weight`; the hyperparameters, early-stopping
    rule, evaluation metric and threshold selection are deliberately identical, so the
    approaches differ in exactly one thing each.

    Args:
        train_features: Training design matrix.
        train_labels: Binary training labels.
        val_features: Validation design matrix (early stopping and threshold only).
        val_labels: Binary validation labels.
        config: Full pipeline config.
        feature_set: Feature-set name, recorded on the fitted object.
        tag: Mixed into the derived seed so each approach gets its own stream.
        sample_weight: Optional per-row training weights.

    Returns:
        A fitted classifier with `threshold` already chosen on the validation block.
    """
    import lightgbm as lgb

    params = dict(config["wait_model"]["classifier"])
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
    for key in (
        "decision_threshold",
        "threshold_objective",
        "target_recall",
        "min_positives_per_level",
        "acuity_sample_weights",
    ):
        params.pop(key, None)

    model = lgb.LGBMClassifier(
        random_state=derive_seed(config["seed"], f"high_acuity::{tag}"),
        verbose=-1,
        n_jobs=-1,
        **params,
    )
    model.fit(
        train_features,
        np.asarray(train_labels, dtype="int8"),
        sample_weight=sample_weight,
        eval_set=[(val_features, np.asarray(val_labels, dtype="int8"))],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    classifier = ProlongedWaitClassifier(
        model=model,
        feature_names=list(train_features.columns),
        feature_set=feature_set,
        best_iteration=getattr(model, "best_iteration_", None),
    )
    threshold, rule = choose_decision_threshold(
        val_labels, classifier.predict_proba(val_features), config
    )
    classifier.threshold = threshold
    classifier.threshold_source = "val"
    classifier.threshold_objective = rule
    logger.info(
        "Fitted %r classifier: %d trees, pooled threshold %.4f (%s)",
        tag,
        classifier.best_iteration or -1,
        threshold,
        rule,
    )
    return classifier


def compare_high_acuity_approaches(
    blocks: dict[str, Any],
    artifacts: dict[str, Any],
    config: dict[str, Any],
    schema: Schema | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all four approaches on the same split and return one comparison table.

    Every approach is fitted on `train`, has its threshold(s) chosen on `val`, and is
    scored once on `test`. No approach sees the test block before it is scored, and
    every one of them uses the same queue features and the same bootstrap procedure, so
    the four are comparable.

    Args:
        blocks: `BlockData` per block name, from `prepare_blocks`.
        artifacts: The ablation's fitted artifacts, used to reuse the shipped
            classifier as the `pooled` control.
        config: Full pipeline config.
        schema: Data contract, for the leakage assertion on every design matrix.

    Returns:
        Tuple of (one row per approach and scope, a JSON-serialisable summary).

    Raises:
        KeyError: If no fitted classifier is present in `artifacts`.
    """
    feature_set = next(
        (key.split("::")[0] for key in artifacts if key.endswith("::classifier")), None
    )
    if feature_set is None:
        raise KeyError(
            "No fitted classifier in the ablation artifacts, so there is no pooled "
            "control to compare against."
        )

    designs = {
        name: make_design_matrix(blocks[name].frame, feature_set, config, schema)[0]
        for name in ("train", "val", "test")
    }
    levels = {
        name: blocks[name].frame["triage_level"].to_numpy()
        for name in ("train", "val", "test")
    }
    pooled_labels = {
        name: blocks[name].frame["prolonged_wait"].to_numpy(dtype="int8")
        for name in ("train", "val", "test")
    }

    threshold_minutes = float(config["features"]["prolonged_wait_threshold_minutes"])
    targets = acuity_target_minutes(config)
    results: list[ApproachResult] = []

    # ---------------------------------------------------------------- (a) pooled
    shipped = artifacts[f"{feature_set}::classifier"]["classifier"]
    pooled = ApproachResult(
        name="pooled",
        description=(
            "the shipped classifier: one model, one target, with a per-acuity "
            "threshold wherever enough positives exist to fit one"
        ),
        label_definition=f"wait > {threshold_minutes:.0f} min, same for every acuity",
        scope="all acuities",
    )
    _evaluate_scores(
        pooled,
        pooled_labels["val"],
        shipped.predict_proba(designs["val"]),
        levels["val"],
        pooled_labels["test"],
        shipped.predict_proba(designs["test"]),
        levels["test"],
        config,
    )
    results.append(pooled)

    # ---------------------------------------------------- (b) acuity sample weights
    weights, weight_summary = acuity_sample_weights(levels["train"], config)
    weighted_model = _train_classifier(
        designs["train"],
        pooled_labels["train"],
        designs["val"],
        pooled_labels["val"],
        config,
        feature_set,
        tag="acuity_weighted",
        sample_weight=weights,
    )
    weighted = ApproachResult(
        name="acuity_weighted",
        description=(
            "the same model and the same target, with LightGBM sample weights "
            "up-weighting high-acuity rows"
        ),
        label_definition=f"wait > {threshold_minutes:.0f} min, same for every acuity",
        scope="all acuities",
        notes=[f"training weights by acuity: {weight_summary}"],
    )
    _evaluate_scores(
        weighted,
        pooled_labels["val"],
        weighted_model.predict_proba(designs["val"]),
        levels["val"],
        pooled_labels["test"],
        weighted_model.predict_proba(designs["test"]),
        levels["test"],
        config,
    )
    results.append(weighted)

    # -------------------------------------------- (c) a separate ESI 1-2-only model
    high_masks = {
        name: np.isin(levels[name].astype("float64"), HIGH_ACUITY_LEVELS)
        for name in ("train", "val", "test")
    }
    high = ApproachResult(
        name="high_acuity_only",
        description="a separate classifier fitted on ESI 1-2 rows only",
        label_definition=f"wait > {threshold_minutes:.0f} min, same for every acuity",
        scope="ESI 1-2 only",
    )
    train_positives = int(pooled_labels["train"][high_masks["train"]].sum())
    val_positives = int(pooled_labels["val"][high_masks["val"]].sum())
    if train_positives < MIN_POSITIVES_TO_REPORT or val_positives < MIN_POSITIVES_TO_REPORT:
        # Refusing to fit IS the finding. A booster trained on a handful of positives
        # still emits scores, and those scores would look exactly like a result.
        high.notes.append(
            f"NOT FITTED: {train_positives} positive case(s) in train and "
            f"{val_positives} in val, against a minimum of {MIN_POSITIVES_TO_REPORT}. "
            "A model fitted on that many positives cannot be evaluated, and a "
            "threshold chosen from them would be noise."
        )
        logger.warning("high_acuity_only not fitted: %s", high.notes[-1])
    else:
        high_model = _train_classifier(
            designs["train"].loc[high_masks["train"]],
            pooled_labels["train"][high_masks["train"]],
            designs["val"].loc[high_masks["val"]],
            pooled_labels["val"][high_masks["val"]],
            config,
            feature_set,
            tag="high_acuity_only",
        )
        _evaluate_scores(
            high,
            pooled_labels["val"][high_masks["val"]],
            high_model.predict_proba(designs["val"].loc[high_masks["val"]]),
            levels["val"][high_masks["val"]],
            pooled_labels["test"][high_masks["test"]],
            high_model.predict_proba(designs["test"].loc[high_masks["test"]]),
            levels["test"][high_masks["test"]],
            config,
        )
    results.append(high)

    # ------------------------------------------------- (d) acuity-specific target
    acuity_labels = {
        name: add_acuity_target_label(blocks[name].frame, config)["exceeds_acuity_target"]
        .fillna(0)
        .to_numpy(dtype="int8")
        for name in ("train", "val", "test")
    }
    target_model = _train_classifier(
        designs["train"],
        acuity_labels["train"],
        designs["val"],
        acuity_labels["val"],
        config,
        feature_set,
        tag="acuity_target",
    )
    acuity_target = ApproachResult(
        name="acuity_target",
        description=(
            "a different target: did this patient exceed the target wait set for "
            "their own acuity"
        ),
        label_definition=(
            "wait > per-acuity target ("
            + ", ".join(f"ESI {k}: {v:.0f} min" for k, v in sorted(targets.items()))
            + ") -- ASSUMED illustrative values, not clinical standards"
        ),
        scope="all acuities",
        notes=[TARGET_CAVEAT],
    )
    _evaluate_scores(
        acuity_target,
        acuity_labels["val"],
        target_model.predict_proba(designs["val"]),
        levels["val"],
        acuity_labels["test"],
        target_model.predict_proba(designs["test"]),
        levels["test"],
        config,
    )
    results.append(acuity_target)

    table = _flatten(results)
    summary = {
        "ran": True,
        "caveat": TARGET_CAVEAT,
        "feature_set": feature_set,
        "min_positives_to_report": MIN_POSITIVES_TO_REPORT,
        "pooled_threshold_minutes": threshold_minutes,
        "acuity_target_minutes": {str(k): v for k, v in sorted(targets.items())},
        "approaches": {
            result.name: {
                "description": result.description,
                "label_definition": result.label_definition,
                "scope": result.scope,
                "fitted": bool(result.rows),
                "notes": result.notes,
                "thresholds": result.thresholds,
            }
            for result in results
        },
        "rows": table.to_dict(orient="records"),
        "verdict": high_acuity_verdict(table),
    }
    return table, summary


def _flatten(results: list[ApproachResult]) -> pd.DataFrame:
    """Turn the per-approach results into one ordered table.

    Args:
        results: One entry per approach.

    Returns:
        A DataFrame ordered by approach then scope, with the identifying columns first.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in result.rows:
            rows.append(
                {
                    "approach": result.name,
                    "label": result.label_definition,
                    "model_scope": result.scope,
                    **{key: value for key, value in row.items() if key != "approach"},
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    column_order = [
        "approach", "scope", "model_scope", "label", "n", "n_positive",
        "val_positives", "threshold", "threshold_source",
        "recall", "recall_ci_low", "recall_ci_high",
        "fnr", "fnr_ci_low", "fnr_ci_high",
        "precision", "precision_ci_low", "precision_ci_high",
        "tp", "fn", "fp", "reportable", "unreportable_reason",
    ]
    ordered = [column for column in column_order if column in table.columns]
    table = table.loc[:, ordered + [c for c in table.columns if c not in ordered]]

    table["approach"] = pd.Categorical(
        table["approach"], categories=list(APPROACH_ORDER), ordered=True
    )
    table = table.sort_values(["approach", "scope"]).reset_index(drop=True)
    table["approach"] = table["approach"].astype(str)
    return table


def high_acuity_verdict(table: pd.DataFrame) -> dict[str, Any]:
    """State, from the table alone, which approaches gave a reportable ESI 1-2 result.

    Returned as data rather than prose so the report generator can quote it without
    hardcoding a claim that may not hold on the next dataset.

    Args:
        table: The flattened comparison table.

    Returns:
        A dict keyed by high-acuity scope, naming which approaches produced enough
        positive cases to report a recall and what those rates were.
    """
    if table.empty:
        return {"note": "no approach produced any scored rows"}

    verdict: dict[str, Any] = {}
    for scope in ("ESI 1", "ESI 2", "ESI 1-2 pooled"):
        subset = table[table["scope"] == scope]
        if subset.empty:
            continue
        entries = [
            {
                "approach": row["approach"],
                "n_positive": int(row["n_positive"]),
                "reportable": bool(row["reportable"]),
                "recall": row["recall"],
                "fnr": row["fnr"],
                "fnr_ci": [row["fnr_ci_low"], row["fnr_ci_high"]],
                "unreportable_reason": row["unreportable_reason"],
            }
            for _, row in subset.iterrows()
        ]
        reportable = [entry["approach"] for entry in entries if entry["reportable"]]
        verdict[scope] = {
            "approaches": entries,
            "reportable_approaches": reportable,
            "any_reportable": bool(reportable),
        }
    return verdict
