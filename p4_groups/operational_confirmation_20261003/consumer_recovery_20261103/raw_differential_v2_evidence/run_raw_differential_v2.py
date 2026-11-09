"""Two-design differential with D152 acquired in a fresh child process."""
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
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
RUN = HERE / 'raw_differential_v2'
CARRIER = HERE / 'lossless_carriers'
PROTOCOL = HERE / 'differential_protocol_v2.json'
REFERENCE = BASE / 'support_modal_20261009_v2' / 'shards'
D7_SOURCE = HERE / 'raw_differential_v1' / 'candidate_shards' / 'D000007.npz'
DESIGNS = ('D000007', 'D000152')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def proof(path):
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


def imports():
    sys.path[:0] = [str(ROOT / 'src'), str(BASE), str(HERE)]
    import raw_modal_data as D
    import experiment_support_modal_v2 as S
    import raw_only_acquisition as A
    if D.E is not S or A.S is not S:
        raise RuntimeError('native module identity mismatch')
    return D, S, A


def preflight(p):
    errors = []
    def check(label, fn):
        try:
            fn()
        except Exception as exc:
            errors.append({'check': label, 'type': type(exc).__name__, 'message': str(exc)})
    for relative, expected in p['files'].items():
        def verify(relative=relative, expected=expected):
            if sha(ROOT / relative) != expected:
                raise RuntimeError(f'fingerprint mismatch: {relative}')
        check(relative, verify)
    def runner():
        if sha(__file__) != p['runner_sha256']:
            raise RuntimeError('runner fingerprint mismatch')
    check('runner', runner)
    def carriers():
        manifest = json.loads((CARRIER / 'member_manifest.json').read_text())
        for name, expected in manifest['carriers'].items():
            check_file(CARRIER / name, expected)
        for item in manifest['npy_member_comparisons']:
            with zipfile.ZipFile(CARRIER / item['carrier']) as archive:
                data = archive.read(item['member'])
            expected = item['carrier_digest']
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256'] or expected != item['source_digest'] or item['equal'] is not True:
                raise RuntimeError(f"carrier member mismatch: {item['carrier']}/{item['member']}")
        for item in manifest['audit_comparisons']:
            if item['equal'] is not True or item['carrier_digest'] != item['source_digest']:
                raise RuntimeError(f"audit mismatch: {item['carrier']}")
            check_file(CARRIER / item['carrier'], item['carrier_digest'])
    check('carrier members', carriers)
    for did, expected in p['references'].items():
        check(did, lambda did=did, expected=expected: check_file(REFERENCE / f'{did}.npz', expected))
    check('v1 D7 candidate', lambda: check_file(D7_SOURCE, p['v1_d7_candidate']))
    def namespace():
        if RUN.exists():
            raise FileExistsError(f'refusing existing namespace: {RUN}')
    check('run namespace', namespace)
    modules = None
    try:
        modules = imports()
    except Exception as exc:
        errors.append({'check': 'native imports', 'type': type(exc).__name__, 'message': str(exc)})
    if modules:
        check('native contract', modules[1].contract)
        check('native inputs and fingerprint', modules[1].inputs)
    return errors, modules


def child(p):
    import numpy as np
    D, S, A = imports()
    start = time.monotonic()
    state = {'status': 'running', 'design': 'D000152'}
    source = RUN / 'candidate_shards' / 'D000152.npz'
    if source.exists() or (RUN / 'child_checkpoint.json').exists():
        raise FileExistsError('refusing existing D152 child artifacts')
    try:
        if sha(__file__) != p['runner_sha256']:
            raise RuntimeError('runner fingerprint mismatch')
        S.contract()
        d, fp = S.inputs()
        old_out = S.M.G.O.OUT
        signature = S.M.G.O.signature()
        try:
            result = A.acquire_raw_only('D000152', d, fp, CARRIER / 'contexts')
        finally:
            if S.M.G.O.OUT != old_out or S.M.G.O.signature() != signature:
                raise RuntimeError('native context or signature not restored after D152')
        provenance = {key: S.clean_json(result[key]) for key in ('identity', 'audit', 'validation', 'artifact_kind')}
        S.save_npz_atomic(source, raw=result['raw'], command=result['command'],
                          records=json.dumps(S.clean_json(result['physical_records'])),
                          artifact_kind=result['artifact_kind'], provenance=json.dumps(provenance, allow_nan=False))
        state.update(status='candidate_persisted', candidate={'path': str(source), **proof(source)},
                     elapsed_seconds=time.monotonic() - start)
        save(RUN / 'child_checkpoint.json', state)
    except BaseException as exc:
        state.update(status='error', error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}, elapsed_seconds=time.monotonic() - start)
        save(RUN / 'child_checkpoint.json', state)
        raise


def payload(array):
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def compare_array(actual, expected):
    import numpy as np
    a, b = payload(actual), payload(expected)
    result = {'candidate': a, 'reference': b, 'equal': a == b}
    if actual.dtype == expected.dtype and actual.shape == expected.shape:
        left = np.ascontiguousarray(actual).view(np.uint8).reshape(actual.size, actual.dtype.itemsize)
        right = np.ascontiguousarray(expected).view(np.uint8).reshape(expected.size, expected.dtype.itemsize)
        result['bitwise_differing_value_count'] = int(np.count_nonzero(np.any(left != right, axis=1)))
        if np.issubdtype(actual.dtype, np.number) or np.issubdtype(actual.dtype, np.bool_):
            with np.errstate(invalid='ignore', over='ignore'):
                differences = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                maximum = float(np.max(differences)) if differences.size else 0.0
            result['max_abs_difference'] = maximum if np.isfinite(maximum) else None
    return result


def keys(records):
    return [(int(r['replicate']), str(r['profile'])) for r in records]


def execute(p, modules):
    import numpy as np
    D, S, _ = modules
    started = time.monotonic()
    RUN.mkdir()
    receipt = {'status': 'running', 'scientific_gate_credit': False,
               'original_complete_consumer_identity': 'UNVERIFIED', 'raw_command': {},
               'subset_normalized': {}, 'protocol_sha256': sha(PROTOCOL)}
    def checkpoint(stage):
        receipt['stage'] = stage
        receipt['elapsed_seconds'] = time.monotonic() - started
        save(RUN / 'checkpoint.json', receipt)
    def timeout_handler(signum, frame):
        raise TimeoutError('180-second differential deadline reached')
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, p['timeout_seconds'])
    old_dataset, old_out, old_shards = D.DATASET, D.OUT, S.SHARDS
    try:
        checkpoint('copy_v1_D000007')
        candidate = RUN / 'candidate_shards'
        candidate.mkdir()
        check_file(D7_SOURCE, p['v1_d7_candidate'])
        shutil.copyfile(D7_SOURCE, candidate / 'D000007.npz')
        check_file(candidate / 'D000007.npz', p['v1_d7_candidate'])
        receipt['v1_d7_candidate'] = {'path': str(candidate / 'D000007.npz'), **proof(candidate / 'D000007.npz')}
        checkpoint('acquire_D000152_child')
        # subprocess.run kills and waits for the owned child on timeout or interruption.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child-D000152'],
                       cwd=ROOT, check=True, timeout=p['child_timeout_seconds'])
        child_state = json.loads((RUN / 'child_checkpoint.json').read_text())
        if child_state['status'] != 'candidate_persisted':
            raise RuntimeError('child did not persist candidate')
        child_source = candidate / 'D000152.npz'
        check_file(child_source, {k: child_state['candidate'][k] for k in ('bytes', 'sha256')})
        receipt['child_candidate'] = child_state
        checkpoint('compare_raw_sources')
        for did in DESIGNS:
            with np.load(candidate / f'{did}.npz', allow_pickle=False) as actual, np.load(REFERENCE / f'{did}.npz', allow_pickle=False) as expected:
                actual_keys = keys(json.loads(str(actual['records'])))
                expected_keys = keys(json.loads(str(expected['records'])))
                comparisons = {field: compare_array(actual[field], expected[field]) for field in ('raw', 'command')}
                comparisons['physical_keys_equal'] = actual_keys == expected_keys
                receipt['raw_command'][did] = comparisons
                checkpoint('compared_' + did)
                if not comparisons['physical_keys_equal'] or not all(comparisons[field]['equal'] for field in ('raw', 'command')):
                    raise RuntimeError(f'{did} raw, command, or ordered physical keys differ')
        D.DATASET = CARRIER / 'manifest_only.npz'
        records, times, _, _ = D._load_manifest()
        selected = [r for r in records if r['design'] in DESIGNS]
        native_p = S.contract()
        if set(D._source_designs(selected)) != set(DESIGNS) or 'D000007' not in native_p['split']['train'] or 'D000152' not in S.designated_additional(native_p):
            raise RuntimeError('subset is not the two designated training designs')
        if set(DESIGNS) & (set(native_p['split']['validation']) | set(native_p['split']['held'])) or any(r.get('split') not in ('train', 'additional_train') for r in selected):
            raise RuntimeError('subset contains non-training records')
        subset = RUN / 'subset_manifest.npz'
        S.save_npz_atomic(subset, records=json.dumps(selected), times=times)
        D.DATASET = subset
        loaded, loaded_times, _, _ = D._load_manifest()
        if loaded != selected or not np.array_equal(loaded_times, times):
            raise RuntimeError('subset manifest values changed')
        receipt['subset_manifest'] = {'path': str(subset), 'sha256': sha(subset), 'rows': len(selected), 'source': str(CARRIER / 'manifest_only.npz')}
        for label, shards in (('reference', old_shards), ('candidate', candidate)):
            checkpoint('normalize_' + label)
            S.SHARDS = shards
            D.OUT = RUN / (label + '_normalized')
            maps = D.acquire()
            del maps
            maps = D.load_measured()
            del maps
            receipt['subset_normalized'][label] = {'metadata_path': str(D.OUT / 'acquisition.json'), 'metadata_sha256': sha(D.OUT / 'acquisition.json'), 'artifacts': {name: sha(D.OUT / (name + '.npy')) for name in ('waveforms', 'admissible')}}
        for name in ('waveforms', 'admissible'):
            if receipt['subset_normalized']['reference']['artifacts'][name] != receipt['subset_normalized']['candidate']['artifacts'][name]:
                raise RuntimeError(f'normalized subset {name} mismatch')
        receipt['subset_normalized']['equal'] = True
        if sha(ROOT / 'P4_PROGRESS.json') != p['files']['P4_PROGRESS.json']:
            raise RuntimeError('protected progress changed')
        receipt['status'] = 'differential_pass'
    except BaseException as exc:
        receipt['status'] = 'error'
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        child_checkpoint = RUN / 'child_checkpoint.json'
        if child_checkpoint.exists():
            receipt['child_checkpoint'] = json.loads(child_checkpoint.read_text())
    finally:
        D.DATASET, D.OUT, S.SHARDS = old_dataset, old_out, old_shards
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        receipt['elapsed_seconds'] = time.monotonic() - started
        save(RUN / 'receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'receipt': str(RUN / 'receipt.json')}), flush=True)
    if receipt['status'] != 'differential_pass':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--child-D000152', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    p = json.loads(PROTOCOL.read_text())
    if args.child_D000152:
        child(p)
        return
    errors, modules = preflight(p)
    print(json.dumps({'status': 'preflight_failed' if errors else 'preflight_pass', 'errors': errors,
                      'execute_requested': args.execute, 'scientific_gate_credit': False}), flush=True)
    if errors:
        raise SystemExit(1)
    if args.execute:
        execute(p, modules)


if __name__ == '__main__':
    main()
