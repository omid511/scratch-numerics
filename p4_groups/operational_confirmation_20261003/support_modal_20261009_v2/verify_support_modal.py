"""Independent support fit, score, hash, and NumPy replay verification.

This file deliberately does not import the production scorer or any metric
helper.  It rebuilds masks, arithmetic summaries, feature scaling, and MLP
forward passes from saved arrays/checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

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


def close(a: Any, b: Any, tolerance: float = 1e-10) -> None:
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=tolerance)


def load_data() -> dict[str, Any]:
    with np.load(DATASET_PATH, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in ("features", "oracle_features", "truth", "observed", "age", "times")}
        data["records"] = json.loads(str(z["records"]))
        data["dataset_protocol_sha256"] = str(z["protocol_sha256"]) if "protocol_sha256" in z else None
    return data


def independent_strata(data: dict[str, Any]) -> dict[str, np.ndarray]:
    truth = data["truth"]
    observed = data["observed"].astype(bool)
    warm = np.broadcast_to(np.arange(truth.shape[1])[None, :] >= 4, truth.shape)
    prev4 = np.zeros_like(truth); prev4[:, 4:] = truth[:, :-4]
    prev2 = np.zeros_like(truth); prev2[:, 2:] = truth[:, :-2]
    span = np.maximum.reduce((prev4, prev2, truth)) - np.minimum.reduce((prev4, prev2, truth))
    boundary = (truth > -0.02) & (truth < 0.05)
    moving = warm & (span >= 0.005)
    dwell = warm & (span <= 1e-10)
    out = dict(all=np.ones_like(observed, dtype=bool), current=observed, gap=~observed,
               boundary=boundary, boundary_current=boundary & observed,
               boundary_gap=boundary & ~observed,
               post_warmup=warm, warmup=~warm,
               distinct_regimes=moving,
               distinct_regimes_boundary=moving & boundary,
               distinct_regimes_current=moving & observed,
               single_regime_dwell=dwell,
               single_regime_dwell_boundary=dwell & boundary,
               weak_regime_change=warm & ~moving & ~dwell,
               farther_stable=truth >= 0.10,
               farther_stable15=truth >= 0.15,
               unstable_exclusion=(truth <= -0.01) & (truth >= -0.05),
               stable_side=(truth >= 0.01) & (truth <= 0.05))
    for low, high in ((-0.02, 0.0), (0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10)):
        out[f"margin_{low}_{high}"] = (truth >= low) & (truth < high)
    return out


def independent_summary(point: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                         designs: np.ndarray) -> dict[str, Any]:
    per_design: dict[str, Any] = {}
    for design in sorted(set(designs[mask.any(axis=1)].tolist())):
        selected = mask & (designs == design)[:, None]
        if selected.any():
            error = point[:, selected] - truth[selected][None, :]
            per_design[design] = dict(
                frames=int(selected.sum()),
                mae_by_seed=np.mean(np.abs(error), axis=1).tolist(),
                signed_error_by_seed=np.mean(error, axis=1).tolist())
    rows = [r["mae_by_seed"] for r in per_design.values()]
    equal_design = np.mean(rows, axis=0) if rows else None

    def event(selected: np.ndarray, condition: np.ndarray) -> dict[str, Any]:
        count = int(selected.sum())
        events = np.count_nonzero(condition & selected[None, ...], axis=(1, 2))
        return dict(frames=count, events_by_seed=events.tolist(),
                    rate_by_seed=(events / count).tolist() if count else None)

    stable = (truth >= 0.01) & (truth <= 0.05)
    unstable = (truth <= -0.01) & (truth >= -0.05)
    return dict(
        frames=int(mask.sum()), designs=len(per_design), per_design=per_design,
        equal_design_mae_by_seed=equal_design.tolist() if equal_design is not None else None,
        equal_design_mae=float(equal_design.mean()) if equal_design is not None else None,
        optimism_over02=event(mask, point - truth[None, ...] > 0.02),
        unstable_positive_point=event(mask & (truth < 0), point > 0),
        stable_wrong_sign=event(mask & stable, point <= 0),
        unstable_exclusion_wrong_sign=event(mask & unstable, point > 0),
    )


def _expected_scaling(features: np.ndarray, observed: np.ndarray, proper: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = features[proper][observed[proper]]
    assert rows.ndim == 2 and rows.shape[1] == 174 and len(rows)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    active = std > 1e-8 * np.maximum(1.0, np.abs(mean))
    scale = np.where(active, std + 1e-8, 1.0)
    return mean, scale, active


def _rebuild_input(features: np.ndarray, observed: np.ndarray, age: np.ndarray,
                   proper: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                   active: np.ndarray) -> np.ndarray:
    expected_mean, expected_scale, expected_active = _expected_scaling(features, observed, proper)
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale, expected_scale, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(active, expected_active)
    normalized = (features - mean) / scale
    normalized[..., ~active] = 0.0
    normalized[~observed] = 0.0
    result = np.concatenate((normalized, observed[..., None].astype(np.float64), age[..., None]), axis=-1)
    assert result.shape[-1] == 176 and np.isfinite(result).all()
    return result.astype(np.float32)


def _numpy_forward(x: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    """Replay FrameMLP using only NumPy matrix operations."""
    # FrameMLP wraps layers under net.{0,2,4}; tolerate a plain Sequential
    # checkpoint as well so replay remains explicit rather than torch-assisted.
    names = (("net.0.weight", "net.0.bias"), ("net.2.weight", "net.2.bias"),
             ("net.4.weight", "net.4.bias"))
    if names[0][0] not in state:
        names = (("0.weight", "0.bias"), ("2.weight", "2.bias"), ("4.weight", "4.bias"))
    value = np.asarray(x, dtype=np.float64)
    for index, (weight_name, bias_name) in enumerate(names):
        weight = state[weight_name].detach().cpu().numpy().astype(np.float64)
        bias = state[bias_name].detach().cpu().numpy().astype(np.float64)
        value = value @ weight.T + bias
        if index < 2:
            value = np.maximum(value, 0.0)
    return value[..., 0]


def _fit_artifact(fit_id: str, seed: int) -> Path:
    return SCRIPT / f"{fit_id}_seed{seed}.npz"


def _checkpoint(fit_id: str, seed: int) -> Path:
    return SCRIPT / f"{fit_id}_seed{seed}.pt"


def _fits(protocol: dict[str, Any]) -> list[str]:
    return ["original_measured", "original_oracle"] + [
        f"fold{fold}_{arm}" for fold in range(len(protocol["evaluation_folds"])) for arm in ARMS
    ]


def _proper_masks(protocol: dict[str, Any], records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    designs = np.asarray([str(r["design"]) for r in records], dtype="U32")
    splits = np.asarray([str(r["split"]) for r in records], dtype="U16")
    noises = np.asarray([str(r["noise"]) for r in records], dtype="U64")
    noise = np.isin(noises, list(protocol["training"]["fitting_noise"]))
    train = set(protocol["split"]["train"])
    validation = (splits == "validation") & noise
    expected: dict[str, np.ndarray] = {"original": np.isin(designs, list(train)) & noise}
    for fold, excluded in enumerate(protocol["evaluation_folds"]):
        ids = train | (set(protocol["additional_designs"]) - set(excluded))
        expected[f"fold{fold}"] = np.isin(designs, list(ids)) & noise
    assert set(designs[expected["original"]]) == train
    assert set(designs[validation]) == set(protocol["split"]["validation"])
    return expected["original"], validation, expected
def _contribution_groups(records: list[dict[str, Any]], proper: np.ndarray,
                         frame_count: int) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, str, str], int] = {}
    for record, keep in zip(records, proper):
        if keep:
            key = (str(record["design"]), str(record["noise"]), str(record["profile"]))
            grouped[key] = grouped.get(key, 0) + 1
    total_frames = int(proper.sum()) * int(frame_count)
    groups = []
    for (design, noise, profile), record_count in sorted(grouped.items()):
        frames = int(record_count) * int(frame_count)
        groups.append(dict(design=design, noise=noise, profile=profile,
                           records=int(record_count), frames=frames,
                           normalized_weight=float(frames / total_frames),
                           per_frame_weight=float(1.0 / total_frames)))
    assert sum(group["frames"] for group in groups) == total_frames
    return groups, total_frames


def _assert_report(expected: dict[str, Any], actual: dict[str, Any], tolerance: float = 1e-10) -> None:
    assert expected["frames"] == actual["frames"]
    assert expected["designs"] == actual["designs"]
    assert set(expected["per_design"]) == set(actual["per_design"])
    for design in expected["per_design"]:
        e, a = expected["per_design"][design], actual["per_design"][design]
        assert e["frames"] == a["frames"]
        close(e["mae_by_seed"], a["mae_by_seed"], tolerance)
        close(e["signed_error_by_seed"], a["signed_error_by_seed"], tolerance)
    close(expected["equal_design_mae_by_seed"], actual["equal_design_mae_by_seed"], tolerance)
    close(expected["equal_design_mae"], actual["equal_design_mae"], tolerance)
    for name in ("optimism_over02", "unstable_positive_point", "stable_wrong_sign", "unstable_exclusion_wrong_sign"):
        assert expected[name]["frames"] == actual[name]["frames"]
        np.testing.assert_array_equal(expected[name]["events_by_seed"], actual[name]["events_by_seed"])
        close(expected[name]["rate_by_seed"], actual[name]["rate_by_seed"], tolerance)


def _delta(expected_a: dict[str, Any], expected_b: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(set(expected_a["per_design"]) & set(expected_b["per_design"]))
    values = {design: (np.asarray(expected_a["per_design"][design]["mae_by_seed"]) -
                       np.asarray(expected_b["per_design"][design]["mae_by_seed"])).tolist() for design in ids}
    flat = np.asarray(list(values.values()), dtype=float)
    return dict(delta_by_design_seed=values,
                mean_delta=float(flat.mean()) if flat.size else None,
                improved_every_design_seed=bool(flat.size and np.all(flat < 0)))


def run() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    for category in ("inputs", "implementation"):
        for relative, digest in protocol.get(category, {}).items():
            candidate = ROOT / relative
            assert candidate.exists() and sha(candidate) == digest
    training = json.loads(TRAINING_PATH.read_text())
    result = json.loads(RESULTS_PATH.read_text())
    data = load_data()
    records = data["records"]
    truth, observed, age = data["truth"], data["observed"], data["age"]
    assert data["features"].dtype == np.float64 and data["oracle_features"].dtype == np.float64
    assert data["features"].shape == (len(records), 124, 174)
    assert data["oracle_features"].shape == data["features"].shape
    assert truth.shape == (len(records), 124) and observed.shape == truth.shape and age.shape == truth.shape
    assert data["times"].shape == (124,)
    assert np.isfinite(data["features"]).all() and np.isfinite(data["oracle_features"]).all()
    assert np.isfinite(truth).all() and np.isfinite(age).all() and np.isfinite(data["times"]).all()
    assert set(training["fit_ids"]) == set(_fits(protocol))
    assert training["fit_count"] == 24
    assert training["dataset_sha256"] == sha(DATASET_PATH)
    assert training["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert training["predictions_sha256"] == sha(PREDICTIONS_PATH)
    assert training["pinned_sha256"]["protocol"] == sha(PROTOCOL_PATH)
    assert training["pinned_sha256"]["dataset"] == sha(DATASET_PATH)
    assert training["pinned_sha256"]["implementation"] == sha(SCRIPT / "train_support_modal.py")
    assert training["pinned_sha256"]["score_implementation"] == sha(SCRIPT / "score_support_modal.py")
    assert training["pinned_sha256"]["verification_implementation"] == sha(SCRIPT / "verify_support_modal.py")
    assert data["dataset_protocol_sha256"] in (None, sha(PROTOCOL_PATH))
    fit_hash_ids = [f"original_{arm}" for arm in ARMS] + [
        f"fold{fold}_{arm}" for fold in range(len(protocol["evaluation_folds"])) for arm in ARMS
    ]
    for fit_id in fit_hash_ids:
        for seed in SEEDS:
            for suffix in ("npz", "pt"):
                name = f"{fit_id}_seed{seed}.{suffix}"
                assert result["artifact_hashes"][name] == sha(SCRIPT / name)
    assert result["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert result["dataset_sha256"] == sha(DATASET_PATH)
    assert result["training_sha256"] == sha(TRAINING_PATH)
    assert result["predictions_sha256"] == sha(PREDICTIONS_PATH)
    assert result["scoring_source_sha256"] == sha(SCRIPT / "score_support_modal.py")

    expected_original, expected_validation, expected_proper = _proper_masks(protocol, records)
    designs = np.asarray([str(r["design"]) for r in records], dtype="U32")
    splits = np.asarray([str(r["split"]) for r in records], dtype="U16")
    additional = set(protocol["additional_designs"])
    # All designated additional copies remain TRAIN; validation/held are never
    # eligible for any proper mask, and each excluded fold is absent in its fit.
    assert set(designs[np.isin(designs, list(additional))]) == additional
    assert set(splits[np.isin(designs, list(additional))]) == {"train"}
    for split_name in ("train", "validation", "held"):
        assert set(designs[splits == split_name]) == set(protocol["split"][split_name]) | (additional if split_name == "train" else set())
    for key, mask in expected_proper.items():
        assert not np.any(mask & (splits != "train"))
        assert set(designs[mask]) == (set(protocol["split"]["train"]) if key == "original" else
                                      set(protocol["split"]["train"]) | (additional - set(protocol["evaluation_folds"][int(key[4:])])) )

    with np.load(PREDICTIONS_PATH, allow_pickle=False) as z:
        aggregate = z["predictions"].copy()
        aggregate_fit_ids = json.loads(str(z["fit_ids"]))
        seeds = tuple(z["seeds"].tolist())
    assert seeds == SEEDS and aggregate.shape == (24, len(records), 124)
    assert len(aggregate_fit_ids) == 24 and len(set(aggregate_fit_ids)) == 24
    expected_aggregate_fit_ids = [f"{fit_id}_seed{seed}" for fit_id in _fits(protocol) for seed in SEEDS]
    assert aggregate_fit_ids == expected_aggregate_fit_ids
    assert result["fit_ids"] == expected_aggregate_fit_ids
    expected_fit_groups = [f"original_{arm}" for arm in ARMS] + [
        f"fold{fold}_{arm}" for fold in range(len(protocol["evaluation_folds"])) for arm in ARMS
    ]
    assert result["fit_groups"] == expected_fit_groups
    assert np.isfinite(aggregate).all()
    aggregate_lookup = {fit_id: aggregate[index] for index, fit_id in enumerate(aggregate_fit_ids)}
    independent_points: dict[str, np.ndarray] = {}
    fit_artifact_checks = 0
    replay_checks = 0
    max_replay_error = 0.0
    for fit_id in _fits(protocol):
        fit_key = fit_id.rsplit("_", 1)[0]
        arm = fit_id.rsplit("_", 1)[1]
        proper = expected_proper[fit_key]
        features = data["features"] if arm == "measured" else data["oracle_features"]
        rows: list[np.ndarray] = []
        for seed_index, seed in enumerate(SEEDS):
            path = _fit_artifact(fit_id, seed)
            checkpoint_path = _checkpoint(fit_id, seed)
            assert path.exists() and checkpoint_path.exists()
            with np.load(path, allow_pickle=False) as z:
                point = z["predictions"].copy()
                row_contributions = z["row_contributions"].copy()
                row_contribution_weights = z["row_contribution_weights"].copy()
                mean, scale, active = z["mean"].copy(), z["scale"].copy(), z["active"].copy()
                saved_proper, saved_validation = z["proper"].copy(), z["validation"].copy()
                history = z["history"].copy()
                metadata = json.loads(str(z["metadata"]))
                assert str(z["protocol_sha256"]) == sha(PROTOCOL_PATH)
                assert str(z["dataset_sha256"]) == sha(DATASET_PATH)
                assert str(z["fit_id"]) == fit_id and str(z["arm"]) == arm
                assert int(z["seed"]) == seed
            np.testing.assert_array_equal(saved_proper, proper)
            np.testing.assert_array_equal(saved_validation, expected_validation)
            assert point.shape == truth.shape and np.isfinite(point).all()
            assert row_contributions.shape == truth.shape
            assert np.isnan(row_contributions[~proper]).all()
            np.testing.assert_allclose(row_contributions[proper], (point[proper] - truth[proper]) ** 2,
                                       rtol=0, atol=2e-6)
            assert row_contribution_weights.shape == truth.shape
            assert np.all(row_contribution_weights[~proper] == 0)
            groups, total_frames = _contribution_groups(records, proper, truth.shape[1])
            np.testing.assert_allclose(row_contribution_weights[proper], 1.0 / total_frames,
                                       rtol=0, atol=1e-12)
            assert metadata["contribution_groups"] == groups
            assert metadata["contribution_total_frames"] == total_frames
            assert metadata["contribution_per_frame_weight"] == float(1.0 / total_frames)
            assert history.ndim == 2 and history.shape[1] == 3 and len(history) == metadata["epochs"]
            assert int(history[-1, 0]) == metadata["epochs"]
            assert metadata["fit_id"] == fit_id and metadata["seed"] == seed
            assert metadata["proper_records"] == int(proper.sum())
            assert metadata["validation_records"] == int(expected_validation.sum())
            expected_input = _rebuild_input(features, observed, age, proper, mean, scale, active)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert checkpoint["fit_id"] == fit_id and checkpoint["seed"] == seed
            assert checkpoint["protocol_sha256"] == sha(PROTOCOL_PATH)
            assert checkpoint["dataset_sha256"] == sha(DATASET_PATH)
            replay = _numpy_forward(expected_input, checkpoint["state"])
            error = float(np.max(np.abs(replay - point)))
            max_replay_error = max(max_replay_error, error)
            assert error <= 2e-6, (fit_id, seed, error)
            fit_artifact_checks += 1
            replay_checks += int(np.prod(point.shape))
            rows.append(point)
            assert np.array_equal(aggregate_lookup[f"{fit_id}_seed{seed}"], point)
        independent_points[fit_id] = np.stack(rows)
    assert fit_artifact_checks == 24

    # Verify every score denominator, per-design/per-seed arithmetic, profile
    # masks, and safety event count without importing score_support_modal.
    strata = independent_strata(data)
    metrics = result["metrics"]
    profiles = result["profiles"]
    guardrails = result["guardrails"]
    pair_checks = 0
    metric_checks = 0
    expected_conditions: set[str] = set()
    profile_specs = (
        ("cross", ("cross",)), ("stable", ("stable",)), ("retreat", ("retreat",)),
        ("farther_stable", ("farther_stable",)), ("unstable_exclusion", ("unstable_exclusion",)),
        ("original3", ("cross", "stable", "retreat")),
        ("new2", ("farther_stable", "unstable_exclusion")),
    )
    for fold, excluded in enumerate(protocol["evaluation_folds"]):
        fit_ids = [f"original_{arm}" for arm in ARMS] + [f"fold{fold}_{arm}" for arm in ARMS]
        points = {fit_id: independent_points[fit_id] for fit_id in fit_ids}
        for population_name, ids in (("evaluation", excluded), ("secondary_held", protocol["split"]["held"])):
            id_set = set(ids)
            noises = sorted({str(r["noise"]) for r in records if r["design"] in id_set})
            for noise in noises:
                condition = f"fold{fold}/{population_name}/{noise}"
                expected_conditions.add(condition)
                population = np.asarray([r["design"] in id_set and r["noise"] == noise for r in records], dtype=bool)[:, None]
                assert condition in metrics and condition in profiles and condition in guardrails
                for fit_id, point in points.items():
                    for name, mask in strata.items():
                        actual = metrics[condition][fit_id][name]
                        _assert_report(independent_summary(point, truth, population & mask, designs), actual)
                        metric_checks += 1
                    profile_rows = np.asarray([r["profile"] for r in records], dtype="U64")
                    for profile, members in profile_specs:
                        profile_rows_mask = np.isin(profile_rows, members)
                        if not np.any(population[:, 0] & profile_rows_mask):
                            continue
                        profile_mask = population & profile_rows_mask[:, None]
                        expected_all = independent_summary(point, truth, profile_mask & strata["all"], designs)
                        _assert_report(expected_all, profiles[condition][fit_id][f"profile_{profile}"])
                        _assert_report(independent_summary(point, truth, profile_mask & strata["boundary"], designs),
                                       profiles[condition][fit_id][f"boundary_profile_{profile}"])
                        _assert_report(independent_summary(point, truth, profile_mask & strata["current"], designs),
                                       profiles[condition][fit_id][f"current_profile_{profile}"])
                        _assert_report(independent_summary(point, truth, profile_mask & strata["gap"], designs),
                                       profiles[condition][fit_id][f"gap_profile_{profile}"])
                        _assert_report(independent_summary(point, truth, profile_mask & strata["boundary_current"], designs),
                                       profiles[condition][fit_id][f"boundary_current_profile_{profile}"])
                    for name in ("all", "current", "gap", "boundary", "boundary_current", "farther_stable", "unstable_exclusion"):
                        report = metrics[condition][fit_id][name]
                        guard = guardrails[condition][fit_id][name]
                        assert guard["frames"] == report["frames"]
                        close(guard["equal_design_mae"], report["equal_design_mae"])
                        close(guard["optimism_over02_rate_by_seed"], report["optimism_over02"]["rate_by_seed"])
                        close(guard["stable_wrong_sign_rate_by_seed"], report["stable_wrong_sign"]["rate_by_seed"])
                        close(guard["unstable_exclusion_wrong_sign_rate_by_seed"], report["unstable_exclusion_wrong_sign"]["rate_by_seed"])
                # Check every factorial contrast's per-design/per-seed values.
                for name in ("all", "current", "gap", "boundary", "boundary_current", "farther_stable", "unstable_exclusion"):
                    paired = result["paired"][condition][name]
                    expected_measured = _delta(metrics[condition][f"fold{fold}_measured"][name], metrics[condition]["original_measured"][name])
                    expected_oracle = _delta(metrics[condition][f"fold{fold}_oracle"][name], metrics[condition]["original_oracle"][name])
                    expected_om = _delta(metrics[condition]["original_oracle"][name], metrics[condition]["original_measured"][name])
                    expected_em = _delta(metrics[condition][f"fold{fold}_oracle"][name], metrics[condition][f"fold{fold}_measured"][name])
                    for key, expected in (("expanded_minus_original_measured", expected_measured),
                                          ("expanded_minus_original_oracle", expected_oracle),
                                          ("oracle_minus_measured_original", expected_om),
                                          ("oracle_minus_measured_expanded", expected_em)):
                        actual = paired[key]
                        assert actual["delta_by_design_seed"] == expected["delta_by_design_seed"]
                        close(actual["mean_delta"], expected["mean_delta"])
                        assert actual["improved_every_design_seed"] == expected["improved_every_design_seed"]
                    interaction = {}
                    ids_common = set(metrics[condition][f"fold{fold}_measured"][name]["per_design"])
                    ids_common &= set(metrics[condition][f"fold{fold}_oracle"][name]["per_design"])
                    ids_common &= set(metrics[condition]["original_measured"][name]["per_design"])
                    ids_common &= set(metrics[condition]["original_oracle"][name]["per_design"])
                    for design in sorted(ids_common):
                        interaction[design] = (
                            np.asarray(metrics[condition][f"fold{fold}_oracle"][name]["per_design"][design]["mae_by_seed"])
                            - np.asarray(metrics[condition][f"fold{fold}_measured"][name]["per_design"][design]["mae_by_seed"])
                            - np.asarray(metrics[condition]["original_oracle"][name]["per_design"][design]["mae_by_seed"])
                            + np.asarray(metrics[condition]["original_measured"][name]["per_design"][design]["mae_by_seed"])
                        ).tolist()
                    actual_i = paired["support_x_content_interaction"]
                    assert actual_i["interaction_by_design_seed"] == interaction
                    values = np.asarray(list(interaction.values()), dtype=float)
                    close(actual_i["mean_interaction"], float(values.mean()) if values.size else None)
                    assert actual_i["designs"] == len(interaction) and actual_i["seeds"] == 3
                    pair_checks += 1
    assert set(metrics) == expected_conditions
    assert set(profiles) == expected_conditions and set(guardrails) == expected_conditions
    assert result["profile_aggregate_strata"] == [
        "profile_original3", "boundary_profile_original3", "profile_new2", "boundary_profile_new2"
    ]
    assert set(result["score_strata"]) >= {"all", "current", "gap", "boundary", "boundary_current"}
    assert result["profile_groups"]["original"] == ["cross", "stable", "retreat"]
    assert result["profile_groups"]["new"] == ["farther_stable", "unstable_exclusion"]
    stress_present = [condition for condition in expected_conditions if condition.endswith("/iid5") or condition.endswith("/correlated5")]
    assert stress_present, "5% stress conditions are absent from score"
    assert result["no_seed_prediction_averaging"] is True

    verification = dict(
        results_sha256=sha(RESULTS_PATH), dataset_sha256=sha(DATASET_PATH),
        predictions_sha256=sha(PREDICTIONS_PATH), protocol_sha256=sha(PROTOCOL_PATH),
        verification_source_sha256=sha(Path(__file__)), training_source_sha256=sha(SCRIPT / "train_support_modal.py"),
        score_source_sha256=sha(SCRIPT / "score_support_modal.py"), fit_artifact_checks=fit_artifact_checks,
        contribution_metadata_verified=True,
        independent_saved_network_frames=replay_checks, maximum_saved_network_prediction_error=max_replay_error,
        independent_metric_rows=metric_checks, independent_factorial_contrast_rows=pair_checks,
        conditions=len(expected_conditions), seeds=list(SEEDS), fit_count=24,
        original_train_designs=protocol["split"]["train"], validation_designs=protocol["split"]["validation"],
        held_designs=protocol["split"]["held"], additional_designs=protocol["additional_designs"],
        excluded_fold_checks=True, validation_and_held_excluded_from_proper=True,
        exact_hashes_and_denominators=True, no_production_metric_helper_imported=True,
        limitation="Numerical/development verification only; grouped TRAIN folds and old held designs are not blind qualification.",
    )
    (SCRIPT / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    print("SUPPORT_INDEPENDENTLY_VERIFIED", json.dumps(verification), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("verify",))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run()
