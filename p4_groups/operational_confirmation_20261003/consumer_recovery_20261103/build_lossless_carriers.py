#!/usr/bin/env python3
"""Construct byte-preserving, minimal carriers for the frozen consumer inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
WORKSPACE = BASE.parent.parent
ROOT = HERE / "lossless_carriers"
ARCHIVE = HERE / "lossless_carriers.tar.gz"
RECEIPT = HERE / "carrier_bundle_receipt.json"
INPUT_MANIFEST = HERE / "required_input_manifest.json"


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def file_digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def main():
    manifest = json.loads(INPUT_MANIFEST.read_text())
    designs = manifest["designs"]
    if len(designs) != 20 or len(set(designs)) != 20:
        raise ValueError("top-level design roster must contain 20 unique IDs")
    if any(not isinstance(d, str) or not d.isalnum() for d in designs):
        raise ValueError("invalid design ID")
    sources = [BASE / "support_modal_20261009_v2/dataset.npz"]
    for design in designs:
        sources.extend([BASE / f"o1_corrected/trajectories_{design}.npz",
                        BASE / f"o1_corrected/audit_{design}.json"])
    references = {}
    for path in sources:
        rel = path.relative_to(WORKSPACE).as_posix()
        actual = file_digest(path)
        expected = manifest["files"][rel]
        if actual != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            raise ValueError(f"source file bytes/SHA mismatch: {rel}")
        references[rel] = actual

    ROOT.mkdir(exist_ok=True)
    (ROOT / "contexts").mkdir(exist_ok=True)
    comparisons = []
    carriers = {}

    def copy_members(source, destination, members):
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as carrier:
            for member in members:
                data = original.read(member)
                info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                carrier.writestr(info, data)
        with zipfile.ZipFile(destination) as carrier, zipfile.ZipFile(source) as original:
            for member in members:
                before = original.read(member)
                after = carrier.read(member)
                equal = before == after
                if not equal:
                    raise ValueError(f"member mismatch: {source}:{member}")
                comparisons.append({"source": source.relative_to(WORKSPACE).as_posix(),
                                    "carrier": destination.relative_to(ROOT).as_posix(),
                                    "member": member, "source_digest": digest(before),
                                    "carrier_digest": digest(after), "equal": equal})
        carriers[destination.relative_to(ROOT).as_posix()] = file_digest(destination)

    copy_members(sources[0], ROOT / "manifest_only.npz", ["records.npy", "times.npy"])
    audits = []
    for index, design in enumerate(designs):
        trajectory, audit = sources[1 + 2 * index:3 + 2 * index]
        copy_members(trajectory, ROOT / f"contexts/trajectories_{design}.npz", ["identity.npy"])
        target = ROOT / f"contexts/audit_{design}.json"
        before = audit.read_bytes()
        target.write_bytes(before)
        after = target.read_bytes()
        if before != after:
            raise ValueError(f"audit mismatch: {audit}")
        audits.append({"source": audit.relative_to(WORKSPACE).as_posix(),
                       "carrier": target.relative_to(ROOT).as_posix(),
                       "source_digest": digest(before), "carrier_digest": digest(after), "equal": True})
        carriers[target.relative_to(ROOT).as_posix()] = file_digest(target)

    with np.load(ROOT / "manifest_only.npz", allow_pickle=False) as data:
        records = json.loads(data["records"].item())
        times = data["times"]
        if len(records) != 2094 or len(times) != 124:
            raise ValueError(f"native manifest lengths: records={len(records)}, times={len(times)}")
    for design in designs:
        with np.load(ROOT / f"contexts/trajectories_{design}.npz", allow_pickle=False) as data:
            _ = data["identity"]
    if len(comparisons) != 22 or len(audits) != 20:
        raise ValueError(f"comparison counts: NPY={len(comparisons)}, audit={len(audits)}")
    output = {"input_manifest": INPUT_MANIFEST.relative_to(WORKSPACE).as_posix(),
              "source_files": references, "designs": designs, "carriers": carriers,
              "npy_member_comparisons": comparisons, "audit_comparisons": audits,
              "native_load": {"records": len(records), "times": len(times), "identities": len(designs)}}
    (ROOT / "member_manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    expected = set(carriers) | {"member_manifest.json"}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    if len(expected) != 42 or actual != expected:
        raise ValueError(f"package membership mismatch: count={len(actual)}, missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    with ARCHIVE.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        for name in sorted(actual):
            path = ROOT / name
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        names = archive.getnames()
        if len(names) != 42 or set(names) != expected:
            raise ValueError("archive membership mismatch")
    receipt = {"archive": ARCHIVE.relative_to(WORKSPACE).as_posix(), **file_digest(ARCHIVE),
               "source_roster_count": len(designs), "npy_comparisons": len(comparisons),
               "audit_comparisons": len(audits), "package_file_count": len(actual),
               "native_records": len(records), "native_times": len(times), "archive_membership_equal": True}
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
