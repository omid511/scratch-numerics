#!/usr/bin/env python3
"""Send the frozen recovery inputs; execution is reserved for the parent."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = Path(__file__).resolve().parent / "carrier_send_receipt.json"
HOST = "cachyos-1"
REMOTE_PYTHON = "/home/omid5/.local/share/uv-python-recovered/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
FILES = ("lossless_carriers.tar.gz", "raw_only_acquisition.py", "run_raw_differential.py", "differential_protocol.json")
ARCHIVE_SHA256 = "7864949dd5da62e1d0ebe445100fdb42904aaf1f1bbae37a45c468000fb96618"

# This program receives only the JSON packet on stdin. Its output contains no file data.
REMOTE = r'''import base64, hashlib, io, json, os, pathlib, stat, sys, tarfile
DEST = pathlib.Path("/home/omid5/mechanics-recovery-20261102/p4_groups/operational_confirmation_20261003/consumer_recovery_20261103")
FILES = ("lossless_carriers.tar.gz", "raw_only_acquisition.py", "run_raw_differential.py", "differential_protocol.json")
ARCHIVE_SHA = "7864949dd5da62e1d0ebe445100fdb42904aaf1f1bbae37a45c468000fb96618"
DESIGNS = ("D000007", "D000033", "D000068", "D000087", "D000042", "D000045", "D000053", "D000126", "D000062", "D000152", "D000008", "D000146", "D000144", "D000029", "D000020", "D000120", "D000129", "D000114", "D000158", "D000115")
EXPECTED = {"manifest_only.npz", "member_manifest.json"} | {"contexts/trajectories_" + d + ".npz" for d in DESIGNS} | {"contexts/audit_" + d + ".json" for d in DESIGNS}
def proof(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
def check_file(path, data):
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.read_bytes() != data:
        raise ValueError("existing_file_conflict")
def main():
    packet = json.load(sys.stdin)
    if set(packet) != set(FILES):
        raise ValueError("input_membership")
    payload = {}
    for name in FILES:
        item = packet[name]
        data = base64.b64decode(item["base64"], validate=True)
        if proof(data) != {"bytes": item["bytes"], "sha256": item["sha256"]}:
            raise ValueError("input_digest")
        payload[name] = data
    if proof(payload[FILES[0]])["sha256"] != ARCHIVE_SHA:
        raise ValueError("archive_digest")
    with tarfile.open(fileobj=io.BytesIO(payload[FILES[0]]), mode="r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != 42 or set(names) != EXPECTED or any(not m.isfile() for m in members):
            raise ValueError("archive_membership")
        expected_contents = {m.name: tar.extractfile(m).read() for m in members}
        if any(pathlib.PurePosixPath(n).is_absolute() or ".." in pathlib.PurePosixPath(n).parts for n in names):
            raise ValueError("unsafe_archive_path")
        DEST.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            path = DEST / name
            try:
                check_file(path, payload[name])
            except FileNotFoundError:
                pass
        carrier = DEST / "lossless_carriers"
        if carrier.exists() or carrier.is_symlink():
            if not stat.S_ISDIR(carrier.lstat().st_mode):
                raise ValueError("existing_carrier_conflict")
            found = set()
            for path in carrier.rglob("*"):
                mode = path.lstat().st_mode
                if stat.S_ISREG(mode):
                    relative = path.relative_to(carrier).as_posix()
                    found.add(relative)
                    if relative not in expected_contents:
                        raise ValueError("existing_carrier_membership")
                    check_file(path, expected_contents[relative])
                elif not stat.S_ISDIR(mode) or path.relative_to(carrier).as_posix() != "contexts":
                    raise ValueError("existing_carrier_entry")
            if found != EXPECTED:
                raise ValueError("existing_carrier_membership")
        for name in FILES:
            path = DEST / name
            try:
                with path.open("xb") as out:
                    out.write(payload[name])
            except FileExistsError:
                check_file(path, payload[name])
        if not carrier.exists():
            carrier.mkdir()
            tar.extractall(carrier, members=members, filter="data")
        for name in FILES:
            check_file(DEST / name, payload[name])
        for name in EXPECTED:
            check_file(carrier / name, expected_contents[name])
    return {"status": "ok", "files": {name: proof(payload[name]) for name in FILES}, "carrier_file_count": len(EXPECTED), "carrier_verified": True}
try:
    print(json.dumps(main(), sort_keys=True))
except Exception as exc:
    print(json.dumps({"status": "error", "error_class": type(exc).__name__}))
    sys.exit(1)
'''


def make_payload():
    """Build the four-file JSON-compatible packet without contacting the host."""
    packet = {}
    for name in FILES:
        data = (ROOT / name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if name == FILES[0] and sha != ARCHIVE_SHA256:
            raise ValueError("local archive SHA-256 mismatch")
        packet[name] = {"bytes": len(data), "sha256": sha,
                        "base64": base64.b64encode(data).decode("ascii")}
    return packet


def main():
    try:
        receipt_file = RECEIPT.open("x", encoding="utf-8")
    except FileExistsError:
        print(json.dumps({"status": "receipt_exists"}))
        return 1
    with receipt_file:
        receipt_file.write(json.dumps({"returncode": None, "observed_result": {"status": "attempt_incomplete_remote_state_unknown"}}) + "\n")
        receipt_file.flush()
        return run_attempt(receipt_file)


def run_attempt(receipt_file):
    command = ["ssh", "-o", "BatchMode=yes", HOST, shlex.join([REMOTE_PYTHON, "-c", REMOTE])]
    try:
        packet = make_payload()
    except (OSError, ValueError) as exc:
        receipt = {"returncode": None, "observed_result": {"status": "local_input_error", "error_class": type(exc).__name__}}
    else:
        try:
            result = subprocess.run(command, input=json.dumps(packet), text=True,
                                    capture_output=True, timeout=120, check=False)
        except subprocess.TimeoutExpired:
            receipt = {"returncode": None, "observed_result": {"status": "remote_state_unknown"}}
        except OSError as exc:
            receipt = {"returncode": None, "observed_result": {"status": "local_launch_error", "error_class": type(exc).__name__}}
        else:
            try:
                observed = json.loads(result.stdout)
                if not isinstance(observed, dict) or observed.get("status") not in ("ok", "error"):
                    raise ValueError("unexpected remote result")
            except (ValueError, json.JSONDecodeError):
                observed = {"status": "unparseable_remote_result"}
            receipt = {"returncode": result.returncode, "observed_result": observed}
    receipt_file.seek(0)
    receipt_file.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_file.truncate()
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt.get("returncode") == 0 and receipt.get("observed_result", {}).get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
