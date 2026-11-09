"""Score support fits on preregistered additional-fold and held populations.

Metrics follow the campaign's ``S.summary`` convention: a seed-specific
prediction is scored first, then equal-design averaging is applied to the seed
errors.  Seed predictions are never averaged before scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[2]
PROTOCOL_PATH = SCRIPT / "protocol.json"
DATASET_PATH = SCRIPT / "dataset.npz"
TRAINING_PATH = SCRIPT / "training.json"
PREDICTIONS_PATH = SCRIPT / "predictions.npz"
RESULTS_PATH = SCRIPT / "results.json"
SEEDS = (42, 43, 44)
ARMS = ("measured", "oracle")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_data() -> dict[str, Any]:
    with np.load(DATASET_PATH, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in ("features", "oracle_features", "truth", "observed", "age", "times")}
        data["records"] = json.loads(str(z["records"]))
    return data


def strata(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Frozen mask names, including the safety guardrail populations."""
    truth = np.asarray(data["truth"])
    observed = np.asarray(data["observed"], dtype=bool)
    warm = np.broadcast_to(np.arange(truth.shape[1])[None, :] >= 4, truth.shape)
    previous4 = np.zeros_like(truth)
    previous2 = np.zeros_like(truth)
    previous4[:, 4:] = truth[:, :-4]
    previous2[:, 2:] = truth[:, :-2]
    span = np.maximum.reduce((previous4, previous2, truth)) - np.minimum.reduce((previous4, previous2, truth))
    boundary = (truth > -0.02) & (truth < 0.05)
    moving = warm & (span >= 0.005)
    dwell = warm & (span <= 1e-10)
    result = dict(
        all=np.ones_like(observed, dtype=bool),
        current=observed,
        gap=~observed,
        boundary=boundary,
        boundary_current=boundary & observed,
        boundary_gap=boundary & ~observed,
        post_warmup=warm,
        warmup=~warm,
        distinct_regimes=moving,
        distinct_regimes_boundary=moving & boundary,
        distinct_regimes_current=moving & observed,
        single_regime_dwell=dwell,
        single_regime_dwell_boundary=dwell & boundary,
        weak_regime_change=warm & ~moving & ~dwell,
        farther_stable=truth >= 0.10,
        farther_stable15=truth >= 0.15,
        unstable_exclusion=(truth <= -0.01) & (truth >= -0.05),
        stable_side=(truth >= 0.01) & (truth <= 0.05),
    )
    # Retain conventional margin strata for diagnostics.
    for low, high in ((-0.02, 0.0), (0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10)):
        result[f"margin_{low}_{high}"] = (truth >= low) & (truth < high)
    return result


def _event(selected: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    # values has [seed, row, frame]; selected has [row, frame].
    count = int(selected.sum())
    events = np.count_nonzero(values & selected[None, ...], axis=(1, 2))
    return dict(frames=count, events_by_seed=events.tolist(),
                rate_by_seed=(events / count).tolist() if count else None)


def summary(point: np.ndarray, truth: np.ndarray, mask: np.ndarray, designs: np.ndarray) -> dict[str, Any]:
    """Equal-design seed-wise MAE plus preregistered safety events."""
    if point.ndim != 3 or point.shape[1:] != truth.shape:
        raise ValueError(f"point shape {point.shape} does not match truth {truth.shape}")
    per_design: dict[str, Any] = {}
    for design in sorted(set(designs[mask.any(axis=1)].tolist())):
        selected = mask & (designs == design)[:, None]
        if selected.any():
            error = point[:, selected] - truth[selected][None, :]
            per_design[design] = dict(
                frames=int(selected.sum()),
                mae_by_seed=np.mean(np.abs(error), axis=1).tolist(),
                signed_error_by_seed=np.mean(error, axis=1).tolist(),
            )
    rows = [r["mae_by_seed"] for r in per_design.values()]
    equal_design = np.mean(rows, axis=0) if rows else None
    unstable = truth < 0
    stable_side = (truth >= 0.01) & (truth <= 0.05)
    unstable_exclusion = (truth <= -0.01) & (truth >= -0.05)
    return dict(
        frames=int(mask.sum()), designs=len(per_design), per_design=per_design,
        equal_design_mae_by_seed=equal_design.tolist() if equal_design is not None else None,
        equal_design_mae=float(equal_design.mean()) if equal_design is not None else None,
        optimism_over02=_event(mask, point - truth[None, ...] > 0.02),
        unstable_positive_point=_event(mask & unstable, point > 0),
        stable_wrong_sign=_event(mask & stable_side, point <= 0),
        unstable_exclusion_wrong_sign=_event(mask & unstable_exclusion, point > 0),
    )


def _profile_masks(population: np.ndarray, base: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    profiles = np.asarray([str(record["profile"]) for record in records], dtype="U64")
    result: dict[str, np.ndarray] = {}
    for group_name, members in (
        ("original3", ("cross", "stable", "retreat")),
        ("new2", ("farther_stable", "unstable_exclusion")),
    ):
        group_rows = np.isin(profiles, members)
        if group_rows.any():
            result[f"profile_{group_name}"] = population & group_rows[:, None] & base["all"]
            result[f"boundary_profile_{group_name}"] = population & group_rows[:, None] & base["boundary"]
            result[f"current_profile_{group_name}"] = population & group_rows[:, None] & base["current"]
            result[f"gap_profile_{group_name}"] = population & group_rows[:, None] & base["gap"]
            result[f"boundary_current_profile_{group_name}"] = population & group_rows[:, None] & base["boundary_current"]
    for profile in ("cross", "stable", "retreat", "farther_stable", "unstable_exclusion"):
        profile_rows = profiles == profile
        if profile_rows.any():
            result[f"profile_{profile}"] = population & profile_rows[:, None] & base["all"]
            result[f"boundary_profile_{profile}"] = population & profile_rows[:, None] & base["boundary"]
            result[f"current_profile_{profile}"] = population & profile_rows[:, None] & base["current"]
            result[f"gap_profile_{profile}"] = population & profile_rows[:, None] & base["gap"]
            result[f"boundary_current_profile_{profile}"] = population & profile_rows[:, None] & base["boundary_current"]
    return result


def _point_map(protocol: dict[str, Any], aggregate: np.ndarray, fit_ids: list[str], fold: int) -> dict[str, np.ndarray]:
    lookup: dict[str, list[np.ndarray]] = {}
    for index, fit_seed_id in enumerate(fit_ids):
        base, separator, seed_text = fit_seed_id.rpartition("_seed")
        expected_seed = SEEDS[index % len(SEEDS)]
        if not separator or int(seed_text) != expected_seed:
            raise ValueError(f"invalid aggregate fit id/order: {fit_seed_id}")
        lookup.setdefault(base, []).append(aggregate[index])
    lookup_stacked = {base: np.stack(points) for base, points in lookup.items()}
    required = [f"original_{arm}" for arm in ARMS] + [f"fold{fold}_{arm}" for arm in ARMS]
    missing = [fit_id for fit_id in required if fit_id not in lookup_stacked]
    if missing:
        raise ValueError(f"missing support fit artifacts: {missing}")
    for fit_id in required:
        if lookup_stacked[fit_id].shape[0] != len(SEEDS):
            raise ValueError(f"wrong seed count for {fit_id}")
    return {fit_id: lookup_stacked[fit_id] for fit_id in required}


def _design_deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(set(a["per_design"]) & set(b["per_design"]))
    values = {design: (np.asarray(a["per_design"][design]["mae_by_seed"]) -
                       np.asarray(b["per_design"][design]["mae_by_seed"])).tolist() for design in ids}
    signed = {design: (np.asarray(a["per_design"][design]["signed_error_by_seed"]) -
                       np.asarray(b["per_design"][design]["signed_error_by_seed"])).tolist() for design in ids}
    flat = np.asarray(list(values.values()), dtype=float)
    return dict(delta_by_design_seed=values, signed_delta_by_design_seed=signed,
                mean_delta=float(flat.mean()) if flat.size else None,
                improved_every_design_seed=bool(flat.size and np.all(flat < 0)))


def _interaction(metrics: dict[str, dict[str, Any]], original_prefix: str, expanded_prefix: str,
                 stratum: str) -> dict[str, Any]:
    """Compute support × content interaction on per-design/per-seed MAE."""
    em = metrics[f"{expanded_prefix}_measured"][stratum]
    eo = metrics[f"{expanded_prefix}_oracle"][stratum]
    om = metrics[f"{original_prefix}_measured"][stratum]
    oo = metrics[f"{original_prefix}_oracle"][stratum]
    ids = sorted(set(em["per_design"]) & set(eo["per_design"]) & set(om["per_design"]) & set(oo["per_design"]))
    interaction: dict[str, list[float]] = {}
    for design in ids:
        interaction[design] = (
            np.asarray(eo["per_design"][design]["mae_by_seed"])
            - np.asarray(em["per_design"][design]["mae_by_seed"])
            - np.asarray(oo["per_design"][design]["mae_by_seed"])
            + np.asarray(om["per_design"][design]["mae_by_seed"])
        ).tolist()
    values = np.asarray(list(interaction.values()), dtype=float)
    return dict(interaction_by_design_seed=interaction,
                mean_interaction=float(values.mean()) if values.size else None,
                designs=len(interaction), seeds=len(SEEDS))


def score() -> None:
    if RESULTS_PATH.exists():
        raise RuntimeError("support scores are immutable once published")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    data = load_data()
    training = json.loads(TRAINING_PATH.read_text())
    if training["dataset_sha256"] != sha(DATASET_PATH):
        raise RuntimeError("training dataset hash does not match acquired dataset")
    if training["predictions_sha256"] != sha(PREDICTIONS_PATH):
        raise RuntimeError("training prediction hash does not match aggregate predictions")
    with np.load(PREDICTIONS_PATH, allow_pickle=False) as z:
        aggregate = z["predictions"].copy()
        fit_ids = json.loads(str(z["fit_ids"]))
        seeds = z["seeds"].tolist()
    if tuple(seeds) != SEEDS or aggregate.shape[0] != len(fit_ids):
        raise ValueError("prediction seed/fitting index mismatch")
    truth = data["truth"]
    records = data["records"]
    designs = np.asarray([str(r["design"]) for r in records], dtype="U32")
    base_masks = strata(data)
    metrics: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    guardrails: dict[str, Any] = {}
    conditions: list[str] = []
    artifact_hashes: dict[str, str] = {PREDICTIONS_PATH.name: sha(PREDICTIONS_PATH)}
    artifact_fit_ids = [f"original_{arm}" for arm in ARMS] + [
        f"fold{fold}_{arm}" for fold in range(len(protocol["evaluation_folds"])) for arm in ARMS
    ]
    for fit_id in artifact_fit_ids:
        for seed in SEEDS:
            artifact_hashes[f"{fit_id}_seed{seed}.npz"] = sha(SCRIPT / f"{fit_id}_seed{seed}.npz")
            artifact_hashes[f"{fit_id}_seed{seed}.pt"] = sha(SCRIPT / f"{fit_id}_seed{seed}.pt")

    for fold, excluded in enumerate(protocol["evaluation_folds"]):
        point_map = _point_map(protocol, aggregate, fit_ids, fold)
        # Evaluation fold and old held are separate populations.  Each fold's
        # expanded model is retained when scoring the held secondary population.
        for population_name, ids in (("evaluation", excluded), ("secondary_held", protocol["split"]["held"])):
            id_set = set(ids)
            noises = sorted({str(r["noise"]) for r in records if r["design"] in id_set})
            for noise in noises:
                condition = f"fold{fold}/{population_name}/{noise}"
                conditions.append(condition)
                population = np.asarray([r["design"] in id_set and r["noise"] == noise for r in records], dtype=bool)[:, None]
                masks = {name: population & mask for name, mask in base_masks.items()}
                fit_metrics = {
                    fit_id: {name: summary(point, truth, selected, designs) for name, selected in masks.items()}
                    for fit_id, point in point_map.items()
                }
                metrics[condition] = fit_metrics
                profile_masks = _profile_masks(population, base_masks, records)
                profiles[condition] = {
                    fit_id: {name: summary(point, truth, selected, designs) for name, selected in profile_masks.items()}
                    for fit_id, point in point_map.items()
                }
                # Every arm is retained in the safety table, with event rates
                # and denominators copied from arithmetic summaries.
                guardrails[condition] = {
                    fit_id: {
                        name: {
                            "frames": report["frames"],
                            "equal_design_mae": report["equal_design_mae"],
                            "optimism_over02_rate_by_seed": report["optimism_over02"]["rate_by_seed"],
                            "stable_wrong_sign_rate_by_seed": report["stable_wrong_sign"]["rate_by_seed"],
                            "unstable_exclusion_wrong_sign_rate_by_seed": report["unstable_exclusion_wrong_sign"]["rate_by_seed"],
                        }
                        for name, report in fit_metrics[fit_id].items()
                        if name in ("all", "current", "gap", "boundary", "boundary_current", "farther_stable", "unstable_exclusion")
                    }
                    for fit_id in point_map
                }
                # Original-vs-expanded content effects, support effects, and the
                # factorial interaction are all retained per design and seed.
                paired[condition] = {}
                for stratum_name in ("all", "current", "gap", "boundary", "boundary_current", "farther_stable", "unstable_exclusion"):
                    paired[condition][stratum_name] = {
                        "expanded_minus_original_measured": _design_deltas(fit_metrics[f"fold{fold}_measured"][stratum_name], fit_metrics[f"original_measured"][stratum_name]),
                        "expanded_minus_original_oracle": _design_deltas(fit_metrics[f"fold{fold}_oracle"][stratum_name], fit_metrics[f"original_oracle"][stratum_name]),
                        "oracle_minus_measured_original": _design_deltas(fit_metrics[f"original_oracle"][stratum_name], fit_metrics[f"original_measured"][stratum_name]),
                        "oracle_minus_measured_expanded": _design_deltas(fit_metrics[f"fold{fold}_oracle"][stratum_name], fit_metrics[f"fold{fold}_measured"][stratum_name]),
                        "support_x_content_interaction": _interaction(fit_metrics, "original", f"fold{fold}", stratum_name),
                    }
    result = dict(
        scope=protocol.get("scope"), protocol_sha256=sha(PROTOCOL_PATH),
        dataset_sha256=sha(DATASET_PATH), predictions_sha256=sha(PREDICTIONS_PATH),
        training_sha256=sha(TRAINING_PATH),
        scoring_source_sha256=sha(Path(__file__)), fit_ids=fit_ids,
        fit_groups=[f"original_{arm}" for arm in ARMS] + [
            f"fold{fold}_{arm}" for fold in range(len(protocol["evaluation_folds"])) for arm in ARMS
        ],
        profile_aggregate_strata=["profile_original3", "boundary_profile_original3",
                                  "profile_new2", "boundary_profile_new2"],
        metrics=metrics, profiles=profiles, paired=paired, guardrails=guardrails,
        artifact_hashes=artifact_hashes,
        score_strata=["all", "current", "gap", "boundary", "boundary_current", "farther_stable", "unstable_exclusion"],
        safety_events=["optimism_over02", "stable_wrong_sign", "unstable_exclusion_wrong_sign", "unstable_positive_point"],
        stress_noise=["iid5", "correlated5"],
        profile_groups={"original": ["cross", "stable", "retreat"], "new": ["farther_stable", "unstable_exclusion"]},
        no_seed_prediction_averaging=True,
        interpretation="Grouped additional TRAIN folds are primary development evidence; the two original held designs are secondary. Native gap predictions remain unqualified extrapolation.",
    )
    save_json(RESULTS_PATH, result)
    print("SUPPORT_SCORED", len(conditions), "conditions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("score", "smoke"))
    args = parser.parse_args()
    if args.phase == "smoke":
        print("SUPPORT_SCORE_MECHANISM_SMOKE", json.dumps({"strata": sorted(strata({"truth": np.zeros((1, 124)), "observed": np.ones((1, 124), dtype=bool)}) )}))
    else:
        score()
