#!/usr/bin/env python3
"""P2 damage-field model vs P1 COMSOL undamaged plates — sanity check.

The P2 field CVAE was trained on simulated fields with severity 0.02-0.15.
Undamaged P1 COMSOL plates should produce near-zero predicted severity.

Approach (per Main's redirect): the CNN-conditioned heteroscedastic head
requires mode shapes, which the P1 COMSOL run does not provide (only
eigenfrequencies f1..f10). We therefore retrain the *freq_only* field CVAE
(``train_field_cvae``, FreqOnlyEncoder on log-frequencies,
``cond_decoder=False``) on data/p2/fields_cnn_ds.npz and push the COMSOL
frequencies through its posterior-mean prediction:

    c = encoder(log f) ; (mu_z, logvar_z) = posterior(c) ; z = mu_z
    field = decoder(z, zeros(d_c))          # de-conditioned decoder

Known caveats, reported alongside the verdict:
  * Mode-count mismatch: the P2 dataset stores M=6 modes; the COMSOL CSV
    has 10. We feed the first 6 COMSOL frequencies.
  * Distribution shift: COMSOL f1..f6 (931-3947 Hz) sit well below the
    training range (1866-7029 Hz), i.e. this input is out-of-distribution.

Verdict: PASS iff mean predicted severity < 0.05.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

sys.path.insert(0, "src")

from mechanics.p2_inverse_damage.field_pipeline import (
    load_field_dataset,
    train_field_cvae,
)

FIELD_NPZ = "data/p2/fields_cnn_ds.npz"
COMSOL_CSV = "data/p1_part1/lhs_results_master.csv"
PASS_THRESHOLD = 0.05


def load_comsol_log_freqs(path: str, n_modes: int) -> np.ndarray:
    """First ``n_modes`` columns f1..fN of the COMSOL master CSV → log Hz."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    freqs = np.array(
        [[float(r[f"f{i}"]) for i in range(1, n_modes + 1)] for r in rows]
    )
    return np.log(freqs)


def posterior_mean_fields(models: dict, X: np.ndarray) -> np.ndarray:
    """Deterministic posterior-mean field prediction, (n, gy, gx).

    Mirrors the interval predictor of the heteroscedastic variant but for
    the plain CVAE decoder (single sigmoid retention grid).
    """
    encoder, posterior, decoder = (
        models["encoder"], models["posterior"], models["decoder"]
    )
    cond_decoder = models["options"]["cond_decoder"]
    lf = np.asarray(X, dtype=np.float32)
    c = np.asarray(encoder.forward(lf))
    cc = c if cond_decoder else np.zeros_like(c)
    mu_z, _ = posterior(c)
    return np.asarray(decoder.forward(mu_z, cc))


def main() -> None:
    dataset = load_field_dataset(FIELD_NPZ)
    n_modes = dataset["log_freqs"].shape[1]
    print(f"[data] P2 fields dataset: {dataset['fields'].shape[0]} samples, "
          f"grid {dataset['fields'].shape[1:]}, n_modes={n_modes}")

    comsol_lf = load_comsol_log_freqs(COMSOL_CSV, n_modes)
    print(f"[data] COMSOL designs: {comsol_lf.shape[0]} "
          f"(first {n_modes} log-freqs)")

    # Distribution-shift diagnostics (raw frequencies).
    tr_f = dataset["freqs"]
    co_f = np.exp(comsol_lf)
    print("[shift] training freq ranges per mode :",
          np.round(tr_f.min(0), 0), "-", np.round(tr_f.max(0), 0))
    print("[shift] COMSOL  freq ranges per mode :",
          np.round(co_f.min(0), 0), "-", np.round(co_f.max(0), 0))
    ood = ((co_f < tr_f.min(0)) | (co_f > tr_f.max(0))).mean()
    print(f"[shift] fraction of COMSOL mode-frequencies outside the "
          f"training per-mode range: {ood:.2%}")

    print("[fit ] training freq_only CVAE (cond_decoder=False) ...")
    models = train_field_cvae(
        dataset, conditioning="freq_only", cond_decoder=False, seed=0,
    )

    # ── COMSOL predictions ────────────────────────────────────────────
    pred = posterior_mean_fields(models, comsol_lf)      # (100, gy, gx)
    pred_sev = 1.0 - pred.mean(axis=(1, 2))              # per-design severity
    frac_hot = float((pred > 0.9).mean())                # pixel retention<0.1
    # severity of a pixel ≈ 1 - retention; count pixels with severity > 0.1
    frac_sev01 = float((pred < 0.9).mean())

    print("\n=== COMSOL undamaged plates: predicted damage ===")
    print(f"mean predicted severity : {pred_sev.mean():.4f}")
    print(f"std  predicted severity : {pred_sev.std():.4f}")
    print(f"min / max severity      : {pred_sev.min():.4f} / "
          f"{pred_sev.max():.4f}")
    print(f"fraction pixels sev>0.1 : {frac_sev01:.4f} "
          f"(hot-pixel fraction retention<0.9)")

    # ── Training-set reference ────────────────────────────────────────
    tr_sev = dataset["severity"][models["splits"]["train"]]
    print("\n=== Training-set severity distribution ===")
    print(f"mean={tr_sev.mean():.4f} std={tr_sev.std():.4f} "
          f"min={tr_sev.min():.4f} max={tr_sev.max():.4f}")

    # In-distribution calibration reference: predict on training inputs.
    tr_pred = posterior_mean_fields(models, dataset["log_freqs"][models["splits"]["train"]])
    tr_pred_sev = 1.0 - tr_pred.mean(axis=(1, 2))
    print(f"model on TRAIN inputs   : mean={tr_pred_sev.mean():.4f} "
          f"std={tr_pred_sev.std():.4f}")

    # ── Verdict ───────────────────────────────────────────────────────
    mean_sev = float(pred_sev.mean())
    print("\n=== VERDICT ===")
    if mean_sev < PASS_THRESHOLD:
        print(f"PASS: mean predicted severity {mean_sev:.4f} < "
              f"{PASS_THRESHOLD} — undamaged COMSOL plates map to "
              f"near-zero predicted damage.")
    else:
        ratio = mean_sev / max(tr_sev.mean(), 1e-12)
        print(f"FLAG: mean predicted severity {mean_sev:.4f} >= "
              f"{PASS_THRESHOLD}.")
        print(f"  That is {ratio:.1%} of the training-set mean severity "
              f"({tr_sev.mean():.4f}); fraction pixels sev>0.1 = "
              f"{frac_sev01:.3f}.")
        if ood > 0.5:
            print("  Likely driver: severe covariate shift — COMSOL "
                  "eigenfrequencies lie far outside the training "
                  "frequency range, so the encoder operates OOD and the "
                  "posterior mean drifts toward the training-damage "
                  "prior rather than reflecting these plates' true "
                  "(zero) damage.")
        print("  Cross-dataset generalization NOT established by this "
              "check; do not deploy the freq_only field head on P1 "
              "COMSOL designs without reconditioning (matched mode "
              "sets / frequency normalization or shape conditioning).")


if __name__ == "__main__":
    main()
