#!/usr/bin/env python3
"""Stage only the differential v2 inputs; launch is reserved for the parent."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = Path(__file__).resolve().parent / "differential_v2_send_receipt.json"
FILES = ("run_raw_differential_v2.py", "differential_protocol_v2.json")
HOST = "cachyos-1"
REMOTE_PYTHON = "/home/omid5/.local/share/uv-python-recovered/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"

REMOTE = r'''import base64, hashlib, json, pathlib, stat, sys
DEST = pathlib.Path("/home/omid5/mechanics-recovery-20261102/p4_groups/operational_confirmation_20261003/consumer_recovery_20261103")
FILES = ("run_raw_differential_v2.py", "differential_protocol_v2.json")
def proof(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
def check_file(path, data):
    if not stat.S_ISREG(path.lstat().st_mode) or path.read_bytes() != data:
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
    for name in FILES:
        try:
            check_file(DEST / name, payload[name])
        except FileNotFoundError:
            pass
    for name in FILES:
        path = DEST / name
        try:
            with path.open("xb") as output:
                output.write(payload[name])
        except FileExistsError:
            check_file(path, payload[name])
    for name in FILES:
        check_file(DEST / name, payload[name])
    return {"status": "ok", "files": {name: proof(payload[name]) for name in FILES}}
try:
    print(json.dumps(main(), sort_keys=True))
except Exception as exc:
    print(json.dumps({"status": "error", "error_class": type(exc).__name__}))
    sys.exit(1)
'''


def make_payload():
    """Build the two-file JSON-compatible packet without contacting the host."""
    packet = {}
    for name in FILES:
        data = (ROOT / name).read_bytes()
        packet[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
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
