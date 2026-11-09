#!/usr/bin/env python3
"""Explicit, resumable transfer of the frozen dataset; launched only by the parent."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "support_modal_20261009_v2/dataset.npz"
RECEIPTS = Path(__file__).resolve().parent
SIZE = 99114670
SHA = "170f5e783e2fbab6abefce7c4b0a18b54af4d36b345e546ab7c01dcb23154499"
CHUNK = 1024 * 1024
HOST = "cachyos-1"
PYTHON = "/home/omid5/.local/share/uv-python-recovered/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
REMOTE = r'''import base64, fcntl, hashlib, json, os, pathlib, stat, sys
BASE = pathlib.Path("/home/omid5/mechanics-recovery-20261102/p4_groups/operational_confirmation_20261003")
PARTDIR = BASE / "consumer_recovery_20261103/dataset_transfer"
PART = PARTDIR / "dataset.npz.partial"
FINAL = BASE / "support_modal_20261009_v2/dataset.npz"
SIZE = 99114670
SHA = "170f5e783e2fbab6abefce7c4b0a18b54af4d36b345e546ab7c01dcb23154499"
CHUNK = 1048576
EMPTY = hashlib.sha256(b"").hexdigest()
def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("nonregular_file")
def directory(path):
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError("non_directory")
def fingerprint(path):
    regular(path)
    size = 0
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1048576):
            size += len(data)
            h.update(data)
    return {"bytes": size, "sha256": h.hexdigest()}
def main():
    request = json.load(sys.stdin)
    directory(BASE)
    directory(BASE / "consumer_recovery_20261103")
    try:
        PARTDIR.mkdir()
    except FileExistsError:
        directory(PARTDIR)
    lockfd = os.open(PARTDIR / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lockfd).st_mode):
            raise ValueError("invalid_lock")
        fcntl.flock(lockfd, fcntl.LOCK_EX)
        try:
            current = fingerprint(PART)
        except FileNotFoundError:
            current = {"bytes": 0, "sha256": EMPTY}
        try:
            directory(FINAL.parent)
            final = fingerprint(FINAL)
        except FileNotFoundError:
            final = None
        if final is not None and final != {"bytes": SIZE, "sha256": SHA}:
            raise ValueError("final_conflict")
        operation = request["operation"]
        if operation == "status":
            return {"status": "ok", "partial": current, "final": final}
        if operation == "append":
            data = base64.b64decode(request["base64"], validate=True)
            offset = request["offset"]
            if current != {"bytes": offset, "sha256": request["prefix_sha256"]}:
                raise ValueError("partial_conflict")
            if offset < 0 or offset >= SIZE or len(data) != min(CHUNK, SIZE - offset):
                raise ValueError("chunk_length")
            if hashlib.sha256(data).hexdigest() != request["chunk_sha256"]:
                raise ValueError("chunk_digest")
            flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
            if offset == 0:
                try:
                    fd = os.open(PART, flags | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    fd = os.open(PART, flags)
            else:
                fd = os.open(PART, flags)
            with os.fdopen(fd, "ab") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or os.fstat(stream.fileno()).st_size != offset:
                    raise ValueError("nonregular_file")
                stream.write(data)
            return {"status": "ok", "partial": fingerprint(PART), "accepted_bytes": len(data)}
        if operation == "commit":
            if current != {"bytes": SIZE, "sha256": SHA}:
                raise ValueError("partial_incomplete")
            if final is None:
                directory(FINAL.parent)
                try:
                    os.link(PART, FINAL, follow_symlinks=False)
                except FileExistsError:
                    if fingerprint(FINAL) != {"bytes": SIZE, "sha256": SHA}:
                        raise ValueError("final_conflict")
            result = fingerprint(FINAL)
            if result != {"bytes": SIZE, "sha256": SHA}:
                raise ValueError("final_digest")
            return {"status": "ok", "final": result}
        raise ValueError("invalid_operation")
    finally:
        os.close(lockfd)
try:
    print(json.dumps(main(), sort_keys=True))
except Exception as exc:
    print(json.dumps({"status": "error", "error_class": type(exc).__name__}))
    sys.exit(1)
'''


def make_chunk(data, offset, prefix_sha256):
    return {"operation": "append", "offset": offset, "prefix_sha256": prefix_sha256,
            "chunk_sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii")}


def invoke(request, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("attempt budget exhausted")
    command = ["ssh", "-o", "BatchMode=yes", HOST, shlex.join([PYTHON, "-c", REMOTE])]
    start = time.monotonic()
    try:
        result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True,
                                timeout=min(120, remaining), check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError("remote_state_unknown") from exc
    elapsed = time.monotonic() - start
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("remote_state_unknown") from exc
    if result.returncode != 0 or not isinstance(response, dict) or response.get("status") != "ok":
        raise RuntimeError("remote_state_unknown")
    return response, elapsed


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--first-chunk", action="store_true")
    modes.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    deadline = start + (120 if args.first_chunk else 1200)
    receipt = {"attempt": uuid.uuid4().hex, "mode": "first_chunk" if args.first_chunk else "complete",
               "chunks": [], "sent_bytes": 0, "status": "incomplete"}
    try:
        with SOURCE.open("rb") as stream:
            full = hashlib.sha256()
            size = 0
            while data := stream.read(CHUNK):
                full.update(data)
                size += len(data)
            if size != SIZE or full.hexdigest() != SHA:
                raise ValueError("local_dataset_mismatch")
            status, status_elapsed = invoke({"operation": "status"}, deadline)
            receipt["status_elapsed_seconds"] = status_elapsed
            partial = status["partial"]
            offset = partial["bytes"]
            if not isinstance(offset, int) or offset < 0 or offset > SIZE or (offset != SIZE and offset % CHUNK):
                raise ValueError("remote_partial_size")
            if args.first_chunk and offset != 0:
                raise ValueError("first_chunk_requires_empty_partial")
            stream.seek(0)
            prefix = hashlib.sha256()
            remaining = offset
            while remaining:
                data = stream.read(min(CHUNK, remaining))
                if not data:
                    raise ValueError("local_prefix_short")
                prefix.update(data)
                remaining -= len(data)
            if prefix.hexdigest() != partial["sha256"]:
                raise ValueError("remote_prefix_mismatch")
            receipt["initial_partial"] = partial
            if status["final"] is not None and status["final"] != {"bytes": SIZE, "sha256": SHA}:
                raise ValueError("remote_final_mismatch")
            if status["final"] is not None and args.complete:
                receipt["final"] = status["final"]
            else:
                while offset < SIZE:
                    data = stream.read(min(CHUNK, SIZE - offset))
                    if len(data) != min(CHUNK, SIZE - offset):
                        raise ValueError("local_chunk_short")
                    chunk_receipt = {"offset": offset, "bytes": len(data),
                                     "sha256": hashlib.sha256(data).hexdigest(), "confirmed": False}
                    receipt["chunks"].append(chunk_receipt)
                    chunk_start = time.monotonic()
                    try:
                        response, elapsed = invoke(make_chunk(data, offset, prefix.hexdigest()), deadline)
                    finally:
                        chunk_receipt["elapsed_seconds"] = time.monotonic() - chunk_start
                    prefix.update(data)
                    expected = {"bytes": offset + len(data), "sha256": prefix.hexdigest()}
                    if response.get("partial") != expected or response.get("accepted_bytes") != len(data):
                        raise RuntimeError("remote_state_unknown")
                    chunk_receipt["confirmed"] = True
                    receipt["sent_bytes"] += len(data)
                    offset += len(data)
                    if args.first_chunk:
                        break
                if args.complete:
                    response, elapsed = invoke({"operation": "commit"}, deadline)
                    if response.get("final") != {"bytes": SIZE, "sha256": SHA}:
                        raise RuntimeError("remote_state_unknown")
                    receipt["commit_elapsed_seconds"] = elapsed
                    receipt["final"] = response["final"]
            receipt["status"] = "ok"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
        receipt["status"] = "remote_state_unknown" if str(exc) == "remote_state_unknown" else "error"
        receipt["error_class"] = type(exc).__name__
        receipt["reason"] = str(exc) if isinstance(exc, (ValueError, RuntimeError, TimeoutError)) else "local_or_response_error"
    receipt["elapsed_seconds"] = time.monotonic() - start
    path = RECEIPTS / ("dataset_transfer_receipt_" + receipt["attempt"] + ".json")
    with path.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))
    return 0 if receipt["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
