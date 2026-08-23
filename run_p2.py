#!/usr/bin/env python3
"""Proposal-2 inverse damage identification: CLI entry points.

Subcommands:
  generate --n N --grid G --out PATH
      Generate the scalar-damage dataset with the FSDT solver and save an
      .npz archive (frequencies, mode_shapes, damage_factors + provenance JSON).
  train --data PATH [--epochs E]
      Load the .npz dataset, train the autoencoder then the CVAE posterior at
      small scale, and save p2_posterior.pt in the working directory.

NOTE: spatial damage fields are under construction (W1/W2 land them later);
driver v1 covers the scalar pipeline end-to-end.
"""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

DEFAULT_OUT = "data/p2/damage_dataset.npz"
DEFAULT_FIELDS_OUT = "data/p2/field_dataset.npz"
DEFAULT_POSTERIOR = "p2_posterior.pt"
SEED = 42


def P(*a, **kw):
    print(*a, flush=True, **kw)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    import datetime

    import numpy as np

    from mechanics.p2_inverse_damage.damage_data import generate_damage_dataset

    P(f"P2 generate: n={args.n} grid={args.grid}x{args.grid} d_min={args.d_min} seed={args.seed}")
    P("NOTE: spatial damage fields are under construction; this dataset is scalar damage only.")

    dataset = generate_damage_dataset(
        n_samples=args.n,
        grid=(args.grid, args.grid),
        d_min=args.d_min,
        seed=args.seed,
    )

    provenance = {
        "created_at": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "n_samples": args.n,
        "grid": [args.grid, args.grid],
        "d_min": args.d_min,
        "seed": args.seed,
        "damage": "scalar (uniform factor per sample)",
        "numpy_version": np.__version__,
        "notes": "Spatial fields under construction (W1/W2).",
    }

    out = Path(args.out)
    if out.suffix != ".npz":
        out = out.with_suffix(out.suffix + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        frequencies=dataset["frequencies"],
        mode_shapes=dataset["mode_shapes"],
        damage_factors=dataset["damage_factors"],
        provenance=np.array(json.dumps(provenance)),
    )

    P(f"frequencies:   {dataset['frequencies'].shape}")
    P(f"mode_shapes:   {dataset['mode_shapes'].shape}")
    P(f"damage_factors:{dataset['damage_factors'].shape}")
    P(f"Saved {out}")


# ---------------------------------------------------------------------------
# generate-fields
# ---------------------------------------------------------------------------

def cmd_generate_fields(args):
    import datetime

    import numpy as np

    from mechanics.p2_inverse_damage.damage_data import (
        DamageField,
        generate_field_dataset,
        save_dataset_npz,
    )

    P(
        f"P2 generate-fields: n={args.n} field grid={args.gy}x{args.gx} "
        f"M={args.M} N={args.N} seed={args.seed}"
    )
    P(
        "NOTE: each sample costs a damaged quadrature assembly (~1-10 s at "
        "M=N=6..8, 6x6..8x8 field grids); frequencies-only supervision."
    )

    t0 = time.perf_counter()
    dataset = generate_field_dataset(
        n_samples=args.n,
        gy=args.gy,
        gx=args.gx,
        M=args.M,
        N=args.N,
        seed=args.seed,
    )

    out = Path(args.out)
    if out.suffix != ".npz":
        out = out.with_suffix(out.suffix + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        DamageField.from_array(dataset["fields"][i])
        for i in range(args.n)
    ]
    splits = dataset["splits"]
    save_dataset_npz(
        str(out),
        fields=fields,
        measurements={
            "frequencies": dataset["frequencies"],
            "severity": dataset["severity"],
        },
    )

    # Append split indices + provenance alongside the save_dataset_npz payload.
    with np.load(out, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload.update(
        {
            "split_train": np.asarray(splits["train"], dtype=np.int64),
            "split_val": np.asarray(splits["val"], dtype=np.int64),
            "split_test": np.asarray(splits["test"], dtype=np.int64),
            "config": np.array(json.dumps(dataset["config"])),
            "provenance": np.array(
                json.dumps(
                    {
                        "created_at": datetime.datetime.now().isoformat(),
                        "command": " ".join(sys.argv),
                        "n_samples": args.n,
                        "field_grid": [args.gy, args.gx],
                        "M": args.M,
                        "N": args.N,
                        "seed": args.seed,
                        "damage": "spatial retention fields (quadrature assembly)",
                        "numpy_version": np.__version__,
                    }
                )
            ),
        }
    )
    np.savez(out, **payload)

    elapsed = time.perf_counter() - t0
    P(f"fields:      {dataset['fields'].shape}")
    P(f"frequencies: {dataset['frequencies'].shape}")
    P(f"severity:    {dataset['severity'].shape}  mean={dataset['severity'].mean():.4f}")
    P(
        f"splits:      train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )
    P(f"elapsed:     {elapsed:.1f}s ({elapsed / args.n:.2f} s/sample)")
    P(f"Saved {out}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    import numpy as np
    import torch

    from mechanics.p2_inverse_damage.decoder import DamageDecoder
    from mechanics.p2_inverse_damage.encoder import MeasurementEncoder
    from mechanics.p2_inverse_damage.posterior import ConditionalPosterior
    from mechanics.p2_inverse_damage.train import train_autoencoder, train_posterior

    data_path = Path(args.data)
    if data_path.suffix != ".npz":
        data_path = data_path.with_suffix(data_path.suffix + ".npz")
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    with np.load(data_path, allow_pickle=False) as npz:
        freqs = npz["frequencies"]
        modes = npz["mode_shapes"]
        damage = npz["damage_factors"]
        prov = json.loads(str(npz["provenance"])) if "provenance" in npz else {}

    n, n_modes = freqs.shape
    gy, gx = modes.shape[2], modes.shape[3]
    P(f"P2 train: {n} samples, {n_modes} modes, grid {gy}x{gx}, epochs={args.epochs}")
    if prov:
        P(f"dataset generated at {prov.get('created_at', '?')} (seed={prov.get('seed', '?')})")
    P("NOTE: spatial damage fields are under construction; training the scalar-damage pipeline.")

    # Small-scale heads for driver v1: conditioning d_c=32, latent d_z=8.
    # Scalar damage target -> decoder emits a 1x1 field.
    d_c, d_z = 32, 8
    enc = MeasurementEncoder(n_modes=n_modes, grid_size=(gy, gx), d_c=d_c, seed=SEED)
    dec = DamageDecoder(d_z=d_z, d_c=d_c, grid_size=(1, 1), seed=SEED)
    post = ConditionalPosterior(d_z=d_z, d_c=d_c, seed=SEED)

    targets = damage[:, np.newaxis, np.newaxis]  # (n, 1, 1) matches decoder output
    batch_size = min(16, n)

    P(f"[phase 1] autoencoder pre-training ({args.epochs} epochs)...")
    ae_losses = train_autoencoder(
        enc, dec, freqs, modes, targets,
        n_epochs=args.epochs, batch_size=batch_size, seed=SEED,
    )
    P(f"[phase 1] loss {ae_losses[0]:.6f} -> {ae_losses[-1]:.6f}")

    P(f"[phase 2] CVAE posterior ELBO training ({args.epochs} epochs)...")
    post_losses = train_posterior(
        post, enc, dec, freqs, modes, targets,
        n_epochs=args.epochs, batch_size=batch_size, seed=SEED,
    )
    P(f"[phase 2] loss {post_losses[0]:.6f} -> {post_losses[-1]:.6f}")

    ckpt = {
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "posterior": post.state_dict(),
        "config": {
            "n_modes": n_modes, "grid": [gy, gx],
            "d_c": d_c, "d_z": d_z, "epochs": args.epochs,
            "batch_size": batch_size, "seed": SEED,
            "data": str(data_path), "damage": "scalar",
        },
    }
    torch.save(ckpt, DEFAULT_POSTERIOR)
    P(f"Saved {DEFAULT_POSTERIOR}")


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_p2.py",
        description="Proposal-2 inverse damage identification driver.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate the scalar-damage dataset (.npz)")
    p_gen.add_argument("--n", type=int, default=100, help="number of samples (default 100)")
    p_gen.add_argument("--grid", type=int, default=32,
                       help="solver evaluation grid size G (GxG, default 32)")
    p_gen.add_argument("--out", default=DEFAULT_OUT, help=f"output .npz path (default {DEFAULT_OUT})")
    p_gen.add_argument("--d-min", type=float, default=0.3,
                       help="minimum damage factor d in [d_min, 1.0] (default 0.3)")
    p_gen.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 42)")
    p_gen.set_defaults(func=cmd_generate)

    p_train = sub.add_parser("train", help="train autoencoder + posterior on a generated dataset")
    p_train.add_argument("--data", required=True, help="dataset .npz from `generate`")
    p_train.add_argument("--epochs", type=int, default=20,
                         help="epochs per training phase (default 20)")
    p_train.set_defaults(func=cmd_train)

    p_gf = sub.add_parser(
        "generate-fields",
        help="generate the spatial damage-field dataset (.npz, frequencies-only)",
    )
    p_gf.add_argument("--n", type=int, default=100,
                      help="number of samples (default 100)")
    p_gf.add_argument("--gy", type=int, default=6,
                      help="field grid rows (default 6; production 6-8)")
    p_gf.add_argument("--gx", type=int, default=6,
                      help="field grid cols (default 6; production 6-8)")
    p_gf.add_argument("-M", dest="M", type=int, default=6,
                      help="solver basis order M (default 6; production 6-8)")
    p_gf.add_argument("-N", dest="N", type=int, default=6,
                      help="solver basis order N (default 6; production 6-8)")
    p_gf.add_argument("--out", default=DEFAULT_FIELDS_OUT,
                      help=f"output .npz path (default {DEFAULT_FIELDS_OUT})")
    p_gf.add_argument("--seed", type=int, default=SEED,
                      help="RNG seed (default 42)")
    p_gf.set_defaults(func=cmd_generate_fields)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
