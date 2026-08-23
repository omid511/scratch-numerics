"""P2 CNN-conditioning identifiability experiment (three arms, seed 0).

Arms on data/p2/fields_cnn_ds.npz (n=300, 8x8 field grid, M=N=6, seed 42,
full mode shapes on the field grid):
  A. point-MSE CVAE, conditioning='cnn', cond_decoder=False
  B. heteroscedastic Gaussian CVAE head, conditioning='cnn', cond_decoder=False
  C. SP3/SBC gates for both arms via evaluate_sp_gates

References: no-conditioning arms mse ~0.0364-0.0384 at the constant-field
level (~0.0387); direct-regression baseline ~0.1655. Success criterion for
the identifiability question: arm-A val MSE < 0.030.
"""
import json
import sys
import time

import numpy as np

from mechanics.p2_inverse_damage.field_pipeline import (
    evaluate_sp_gates,
    load_field_dataset,
    train_field_cvae,
    train_field_cvae_heteroscedastic,
)

DS = "data/p2/fields_cnn_ds.npz"


def main() -> None:
    dataset = load_field_dataset(DS)
    fields = dataset["fields"]
    val_idx = np.asarray(dataset["splits"]["val"], dtype=int)
    truth = fields[val_idx]

    const_level = float(np.mean((truth - fields[dataset["splits"]["train"]].mean()) ** 2))
    print(f"[ref] constant-field-level MSE (val vs train-mean field): {const_level:.4f}",
          flush=True)

    results: dict = {"constant_field_mse": const_level}

    # ── Arm A: point-MSE CVAE with CNN conditioning ────────────────────
    t0 = time.perf_counter()
    models_a = train_field_cvae(
        dataset, conditioning="cnn", cond_decoder=False, seed=0,
    )
    print(f"[arm A] train done in {time.perf_counter() - t0:.1f}s", flush=True)
    ev_a = evaluate_sp_gates(models_a, dataset, n_samples=50, seed=0)
    results["arm_a_pointmse"] = {
        k: v for k, v in ev_a.items()
        if k not in ("ranks",)
    }
    print("[arm A] point-MSE cnn:", json.dumps(
        {k: v for k, v in results["arm_a_pointmse"].items()}, default=float),
        flush=True)

    # ── Arm B: heteroscedastic head with CNN conditioning ──────────────
    t0 = time.perf_counter()
    models_b = train_field_cvae_heteroscedastic(
        dataset, conditioning="cnn", cond_decoder=False, seed=0,
    )
    print(f"[arm B] train done in {time.perf_counter() - t0:.1f}s", flush=True)
    ev_b = evaluate_sp_gates(models_b, dataset, n_samples=50, seed=0)
    results["arm_b_hetero"] = {
        k: v for k, v in ev_b.items() if k != "ranks"
    }
    print("[arm B] hetero cnn:", json.dumps(
        {k: v for k, v in results["arm_b_hetero"].items()}, default=float),
        flush=True)

    with open("data/p2/cnn_identifiability_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("saved data/p2/cnn_identifiability_results.json", flush=True)


if __name__ == "__main__":
    main()
