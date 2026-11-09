"""Rebuild the complete measured consumer from lossless carriers and raw sources."""
import os
import sys
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import platform
import shutil
import signal
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
RUN = HERE / 'full_consumer_v1'
CARRIER = HERE / 'lossless_carriers'
PROTOCOL = HERE / 'full_consumer_protocol.json'
REUSE = HERE / 'raw_differential_v2' / 'candidate_shards'
PROTOCOL_CONTENT_SHA = 'e9f5a297da4a1310b6342cfdeee6e5f080850e53bf2586b469dfc472985c70d0'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def proof(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def check_file(path, expected):
    if proof(path) != expected:
        raise RuntimeError(f'fingerprint mismatch: {path}')


def save(path, value):
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def protocol():
    p = json.loads(PROTOCOL.read_text())
    core = {k: v for k, v in p.items() if k != 'runner_sha256'}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != PROTOCOL_CONTENT_SHA or sha(__file__) != p['runner_sha256']:
        raise RuntimeError('runner or protocol fingerprint mismatch')
    return p


def imports():
    sys.path[:0] = [str(ROOT / 'src'), str(BASE), str(HERE)]
    import raw_modal_data as D
    import experiment_support_modal_v2 as S
    import raw_only_acquisition as A
    if D.E is not S or A.S is not S:
        raise RuntimeError('native module identity mismatch')
    return D, S, A


def carrier_check():
    manifest = json.loads((CARRIER / 'member_manifest.json').read_text())
    for name, expected in manifest['carriers'].items():
        check_file(CARRIER / name, expected)
    for item in manifest['npy_member_comparisons']:
        with zipfile.ZipFile(CARRIER / item['carrier']) as archive:
            data = archive.read(item['member'])
        expected = item['carrier_digest']
        if (len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']
                or expected != item['source_digest'] or item['equal'] is not True):
            raise RuntimeError(f"carrier member mismatch: {item['carrier']}/{item['member']}")
    for item in manifest['audit_comparisons']:
        if item['equal'] is not True or item['carrier_digest'] != item['source_digest']:
            raise RuntimeError(f"audit mismatch: {item['carrier']}")
        check_file(CARRIER / item['carrier'], item['carrier_digest'])
    if len(manifest['designs']) != 20 or len(set(manifest['designs'])) != 20:
        raise RuntimeError('carrier roster is not 20 unique designs')
    return manifest


def manifest_check(D, p, carrier):
    old = D.DATASET
    try:
        D.DATASET = CARRIER / 'manifest_only.npz'
        records, times, text, rows = D._load_manifest()
        designs = D._source_designs(records)
        if (rows != 2094 or len(times) != 124 or set(designs) != set(carrier['designs'])
                or hashlib.sha256(text.encode('utf-8')).hexdigest() != p['record_order_sha256']):
            raise RuntimeError('full carrier manifest rows, designs, or record order differ')
        return records, designs
    finally:
        D.DATASET = old


def preflight(p):
    errors = []
    def check(label, fn):
        try:
            return fn()
        except Exception as exc:
            errors.append({'check': label, 'type': type(exc).__name__, 'message': str(exc)})
    for relative, expected in p['files'].items():
        def verify(relative=relative, expected=expected):
            if sha(ROOT / relative) != expected:
                raise RuntimeError(f'fingerprint mismatch: {relative}')
        check(relative, verify)
    carrier = check('carrier members', carrier_check)
    for did, expected in p['reused_sources'].items():
        check('reuse ' + did, lambda did=did, expected=expected: check_file(REUSE / f'{did}.npz', expected))
    def namespace():
        if RUN.exists():
            raise FileExistsError(f'refusing existing namespace: {RUN}')
    check('run namespace', namespace)
    def disk():
        if shutil.disk_usage(HERE).free < p['minimum_free_bytes']:
            raise RuntimeError('less than 3 GB free')
    check('disk space', disk)
    modules = check('native imports', imports)
    if modules:
        D, S, _ = modules
        check('native contract', S.contract)
        check('native inputs and fingerprint', S.inputs)
        if carrier:
            check('native full manifest', lambda: manifest_check(D, p, carrier))
    return errors, modules, carrier


def child(did, p):
    D, S, A = imports()
    started = time.monotonic()
    state = {'status': 'running', 'design': did}
    source = RUN / 'candidate_shards' / f'{did}.npz'
    checkpoint = RUN / 'checkpoints' / f'{did}.json'
    if source.exists() or checkpoint.exists():
        raise FileExistsError(f'refusing existing child artifacts: {did}')
    try:
        carrier = carrier_check()
        if did not in carrier['designs'] or did in p['reused_sources']:
            raise ValueError('child design outside remaining roster')
        S.contract()
        d, fp = S.inputs()
        old_out = S.M.G.O.OUT
        signature = S.M.G.O.signature()
        try:
            result = A.acquire_raw_only(did, d, fp, CARRIER / 'contexts')
        finally:
            if S.M.G.O.OUT != old_out or S.M.G.O.signature() != signature:
                raise RuntimeError('native context or signature not restored')
        provenance = {key: S.clean_json(result[key]) for key in ('identity', 'audit', 'validation', 'artifact_kind')}
        S.save_npz_atomic(source, raw=result['raw'], command=result['command'],
                          records=json.dumps(S.clean_json(result['physical_records'])),
                          artifact_kind=result['artifact_kind'], provenance=json.dumps(provenance, allow_nan=False))
        state.update(status='candidate_persisted', candidate={'path': str(source), **proof(source)},
                     provenance=provenance, elapsed_seconds=time.monotonic() - started)
        save(checkpoint, state)
    except BaseException as exc:
        state.update(status='error', error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()},
                     elapsed_seconds=time.monotonic() - started)
        save(checkpoint, state)
        raise


def execute(p, modules, carrier):
    import numpy as np
    D, S, _ = modules
    started = time.monotonic()
    RUN.mkdir()
    receipt = {'status': 'error', 'stage': 'claimed', 'scientific_gate_credit': False,
               'sources': {}, 'expected_source_count': 20, 'actual_hashes': {},
               'host': platform.node(), 'python': sys.executable,
               'carrier_manifest': str(CARRIER / 'member_manifest.json'),
               'manifest': str(CARRIER / 'manifest_only.npz'),
               'lineage': 'Lossless carrier manifest and contexts plus individually acquired or pinned reused raw consumer sources; no original full intermediate archive identity claimed.'}
    old_dataset, old_out, old_shards = D.DATASET, D.OUT, S.SHARDS
    def checkpoint(stage):
        receipt['stage'] = stage
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['complete_source_count'] = len(receipt['sources'])
        save(RUN / 'checkpoint.json', receipt)
    def deadline(signum, frame):
        raise TimeoutError('900-second total deadline reached')
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, p['total_timeout_seconds'])
    try:
        control = RUN / 'control'
        control.mkdir()
        paths = [Path(__file__).resolve(), PROTOCOL, CARRIER / 'member_manifest.json'] + [ROOT / x for x in p['files']]
        snapshots = {}
        for index, path in enumerate(dict.fromkeys(paths)):
            dest = control / f'{index:02d}_{path.name}'
            shutil.copyfile(path, dest)
            check_file(dest, proof(path))
            snapshots[str(path)] = {'snapshot': str(dest), **proof(dest)}
        receipt['control_snapshots'] = snapshots
        receipt['protocol_sha256'] = sha(PROTOCOL)
        receipt['native_inputs_fingerprint_expected'] = p['inputs_fingerprint']
        receipt['carrier_files'] = carrier['carriers']
        checkpoint('control_saved')
        candidate = RUN / 'candidate_shards'
        candidate.mkdir()
        (RUN / 'checkpoints').mkdir()
        (RUN / 'logs').mkdir()
        for did, expected in p['reused_sources'].items():
            source = REUSE / f'{did}.npz'
            check_file(source, expected)
            dest = candidate / f'{did}.npz'
            shutil.copyfile(source, dest)
            check_file(dest, expected)
            receipt['sources'][did] = {'kind': 'reused_v2', 'source': str(source), 'path': str(dest), **proof(dest)}
            checkpoint('reused_' + did)
        for did in carrier['designs']:
            if did in p['reused_sources']:
                continue
            checkpoint('acquire_' + did)
            remaining = p['total_timeout_seconds'] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('total deadline reached before child')
            with (RUN / 'logs' / f'{did}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', did], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=min(p['child_timeout_seconds'], remaining))
            state = json.loads((RUN / 'checkpoints' / f'{did}.json').read_text())
            if state['status'] != 'candidate_persisted':
                raise RuntimeError(f'child did not persist candidate: {did}')
            dest = candidate / f'{did}.npz'
            check_file(dest, {k: state['candidate'][k] for k in ('bytes', 'sha256')})
            receipt['sources'][did] = {'kind': 'fresh_child', **state['candidate'], 'checkpoint': str(RUN / 'checkpoints' / f'{did}.json')}
            checkpoint('persisted_' + did)
        records, designs = manifest_check(D, p, carrier)
        if set(receipt['sources']) != set(designs):
            raise RuntimeError('source coverage mismatch')
        S.SHARDS = candidate
        D.DATASET = CARRIER / 'manifest_only.npz'
        D.OUT = RUN / 'normalized'
        checkpoint('native_acquisition')
        maps = D.acquire()
        del maps
        checkpoint('native_load_measured')
        maps = D.load_measured()
        try:
            for arr, shape, dtype in zip(maps, ((2094, 16, 8001), (2094, 8001)), (np.dtype('float32'), np.dtype('bool'))):
                if not isinstance(arr, np.memmap) or arr.flags.writeable or arr.shape != shape or arr.dtype != dtype:
                    raise RuntimeError('normalized read-only array shape or dtype mismatch')
        finally:
            del maps
        meta_path = D.OUT / 'acquisition.json'
        meta = json.loads(meta_path.read_text())
        if (meta['rows'] != len(records) or not all(meta['acquisition_rows'])
                or meta['row_order']['record_manifest_sha256'] != p['record_order_sha256']
                or set(meta['source_shards']) != set(designs)):
            raise RuntimeError('native metadata row order or source coverage mismatch')
        receipt['native_metadata'] = {'path': str(meta_path), **proof(meta_path), 'rows': meta['rows'],
                                      'frames': meta['frames'], 'source_shards': meta['source_shards']}
        for name in ('waveforms', 'admissible'):
            receipt['actual_hashes'][name] = proof(D.OUT / f'{name}.npy')
        receipt['status'] = ('exact_original_consumer_hashes' if all(
            receipt['actual_hashes'][name]['sha256'] == expected for name, expected in p['expected_artifacts'].items())
            else 'consumer_hash_mismatch')
    except BaseException as exc:
        receipt['status'] = 'error'
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        D.DATASET, D.OUT, S.SHARDS = old_dataset, old_out, old_shards
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        try:
            receipt['progress_after'] = proof(ROOT / 'P4_PROGRESS.json')
            if receipt['progress_after']['sha256'] != p['files']['P4_PROGRESS.json']:
                receipt['status'] = 'error'
                receipt['progress_error'] = 'protected progress changed'
        except Exception as exc:
            receipt['status'] = 'error'
            receipt['progress_error'] = f'{type(exc).__name__}: {exc}'
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['complete_source_count'] = len(receipt['sources'])
        save(RUN / 'receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'receipt': str(RUN / 'receipt.json')}), flush=True)
    if receipt['status'] != 'exact_original_consumer_hashes':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args()
    p = protocol()
    if args.child:
        child(args.child, p)
        return
    errors, modules, carrier = preflight(p)
    print(json.dumps({'status': 'preflight_failed' if errors else 'preflight_pass', 'errors': errors,
                      'execute_requested': args.execute, 'scientific_gate_credit': False}), flush=True)
    if errors:
        raise SystemExit(1)
    if args.execute:
        execute(p, modules, carrier)


if __name__ == '__main__':
    main()
