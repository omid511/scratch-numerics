"""Native raw/command acquisition without archived state replay."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np

import experiment_support_modal_v2 as S


def acquire_raw_only(did, d, fp, context_dir) -> dict:
    """Acquire native consumer-source rows while omitting the state archive."""
    with S.threadpool_limits(limits=1):
        p = S.contract()
        original = set(p["split"]["train"]) | set(p["split"]["validation"]) | set(p["split"]["held"])
        additional = S.designated_additional(p)
        if did not in original and did not in additional:
            raise ValueError(f"design is outside frozen support set: {did}")

        if did in original:
            raw, command, records = S.source_records()
            indices = S._source_physical_for_design(did, raw, records)
            saved_out = S.M.G.O.OUT
            try:
                S.M.G.O.OUT = Path(context_dir)
                identity, plate, audit = S._load_o1_context(d, fp, did)
            finally:
                S.M.G.O.OUT = saved_out
            validation = S.validate_plate(plate, p, audit)

            physical_records = []
            raw_rows = []
            command_rows = []
            for source_index in indices:
                record = copy.deepcopy(records[source_index])
                record["source_index"] = int(source_index)
                raw_rows.append(raw[source_index])
                command_rows.append(command[source_index])
                physical_records.append(record)

            observer = copy.copy(plate)
            observer.reduced = dict(plate.reduced)
            observer.reduced["projection"] = np.vstack((plate.reduced["projection"], np.eye(2 * plate.count)))
            for replicate in range(p["schedule_replicates"]):
                for profile in (profile for profile in p["applicable_profiles"][did] if profile not in S.OLD_PROFILES):
                    description = S.extended_schedule(did, replicate, profile, p)
                    _, response, command_activity = S.T.acquire(observer, description, replicate)
                    raw_rows.append(response[:len(plate.sensor)])
                    command_rows.append(command_activity.astype(bool))
                    physical_records.append(dict(design=did, split=S.split_map(p)[did], replicate=replicate,
                        profile=profile, schedule=description, source_kind="added_regime"))
        else:
            if not __import__("torch").cuda.is_available():
                raise RuntimeError("additional support acquisition requires CUDA; CPU propagation is forbidden")
            saved_out = S.M.G.O.OUT
            try:
                S.M.G.O.OUT = Path(context_dir)
                identity, plate, audit = S._load_o1_context(d, fp, did)
            finally:
                S.M.G.O.OUT = saved_out
            validation = S.validate_plate(plate, p, audit)

            observer = copy.copy(plate)
            observer.reduced = dict(plate.reduced)
            observer.reduced["projection"] = np.vstack((plate.reduced["projection"], np.eye(2 * plate.count)))
            raw_rows, command_rows, physical_records = [], [], []
            for replicate in range(p["schedule_replicates"]):
                for profile in p["applicable_profiles"][did]:
                    description = S.extended_schedule(did, replicate, profile, p)
                    times, response, command_activity = S.T.acquire(observer, description, replicate)
                    if response.shape != (len(plate.sensor) + 2 * plate.count, len(times)):
                        raise RuntimeError(f"unexpected augmented acquisition shape for {did}/{profile}/{replicate}")
                    if not np.isfinite(response).all() or not np.isfinite(command_activity).all():
                        raise RuntimeError(f"nonfinite additional trajectory for {did}/{profile}/{replicate}")
                    raw_rows.append(response[: len(plate.sensor)])
                    command_rows.append(command_activity.astype(bool))
                    physical_records.append(
                        dict(
                            design=did,
                            split="train",
                            replicate=int(replicate),
                            profile=profile,
                            schedule=description,
                            source_kind="additional",
                        )
                    )
                    print("SUPPORT_TRAJECTORY_ACQUIRED", did, replicate, profile, flush=True)

        return {
            "raw": np.stack(raw_rows),
            "command": np.stack(command_rows),
            "physical_records": physical_records,
            "identity": identity,
            "audit": audit,
            "validation": validation,
            "artifact_kind": "raw_consumer_source",
        }
