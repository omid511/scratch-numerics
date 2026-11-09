"""Isolated two-design raw and native-normalization differential; not a gate run."""
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import signal
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
RUN = HERE / 'raw_differential_v1'
CARRIER = HERE / 'lossless_carriers'
PROTOCOL = HERE / 'differential_protocol.json'
DESIGNS = ('D000007', 'D000152')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, proof):
    if path.stat().st_size != proof['bytes'] or sha(path) != proof['sha256']:
        raise RuntimeError(f'fingerprint mismatch: {path}')


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
    check('runner', lambda: None if sha(__file__) == p['runner_sha256'] else (_ for _ in ()).throw(RuntimeError('runner fingerprint mismatch')))
    def carriers():
        manifest = json.loads((CARRIER / 'member_manifest.json').read_text())
        for name, proof in manifest['carriers'].items():
            check_file(CARRIER / name, proof)
        for item in manifest['npy_member_comparisons']:
            with zipfile.ZipFile(CARRIER / item['carrier']) as archive:
                data = archive.read(item['member'])
            proof = item['carrier_digest']
            if len(data) != proof['bytes'] or hashlib.sha256(data).hexdigest() != proof['sha256'] or proof != item['source_digest'] or item['equal'] is not True:
                raise RuntimeError(f"carrier member mismatch: {item['carrier']}/{item['member']}")
        for item in manifest['audit_comparisons']:
            if item['equal'] is not True or item['carrier_digest'] != item['source_digest']:
                raise RuntimeError(f"audit member mismatch: {item['carrier']}")
            check_file(CARRIER / item['carrier'], item['carrier_digest'])
    check('carrier members', carriers)
    for did, proof in p['references'].items():
        check(did, lambda did=did, proof=proof: check_file(BASE / 'support_modal_20261009_v2' / 'shards' / f'{did}.npz', proof))
    check('run namespace', lambda: None if not RUN.exists() else (_ for _ in ()).throw(FileExistsError(f'refusing existing namespace: {RUN}')))
    sys.path[:0] = [str(ROOT / 'src'), str(BASE), str(HERE)]
    modules = {}
    def native():
        import raw_modal_data as D
        import experiment_support_modal_v2 as S
        import raw_only_acquisition as A
        if D.E is not S or A.S is not S:
            raise RuntimeError('native module identity mismatch')
        modules.update(D=D, S=S, A=A)
    check('native imports', native)
    if modules:
        check('native contract', modules['S'].contract)
        def inputs():
            modules['d'], modules['fp'] = modules['S'].inputs()
        check('native inputs and fingerprint', inputs)
    return errors, modules


def payload(array):
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def keys(records):
    return [(int(r['replicate']), str(r['profile'])) for r in records]


def save(path, value):
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def execute(p, modules):
    import numpy as np
    D, S, A = modules['D'], modules['S'], modules['A']
    d, fp = modules['d'], modules['fp']
    started = time.monotonic()
    RUN.mkdir()  # exclusive namespace claim; never reuse a failed run
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
    original_out = S.M.G.O.OUT
    old_dataset, old_out, old_shards = D.DATASET, D.OUT, S.SHARDS
    try:
        checkpoint('acquiring_raw_sources')
        signature = S.M.G.O.signature()
        candidate = RUN / 'candidate_shards'
        candidate.mkdir()
        for did in DESIGNS:
            checkpoint('acquire_' + did)
            try:
                result = A.acquire_raw_only(did, d, fp, CARRIER / 'contexts')
            finally:
                if S.M.G.O.OUT != original_out or S.M.G.O.signature() != signature:
                    raise RuntimeError(f'native context or signature not restored after {did}')
            with np.load(old_shards / f'{did}.npz', allow_pickle=False) as archive:
                reference_records = json.loads(str(archive['records']))
                if keys(result['physical_records']) != keys(reference_records):
                    raise RuntimeError(f'physical record order mismatch: {did}')
                equality = {}
                for field in ('raw', 'command'):
                    actual, expected = payload(result[field]), payload(archive[field])
                    equality[field] = {'candidate': actual, 'reference': expected, 'equal': actual == expected}
                    if actual != expected:
                        receipt['raw_command'][did] = equality
                        raise RuntimeError(f'{did} {field} byte mismatch')
            source = candidate / f'{did}.npz'
            provenance = {key: S.clean_json(result[key]) for key in ('identity', 'audit', 'validation', 'artifact_kind')}
            S.save_npz_atomic(source, raw=result['raw'], command=result['command'],
                              records=json.dumps(S.clean_json(result['physical_records'])),
                              artifact_kind=result['artifact_kind'], provenance=json.dumps(provenance, allow_nan=False))
            receipt['raw_command'][did] = equality
            receipt['raw_command'][did]['physical_keys_equal'] = True
            receipt['raw_command'][did]['candidate_source'] = {'path': str(source), 'sha256': sha(source), 'bytes': source.stat().st_size, 'artifact_kind': result['artifact_kind']}
            checkpoint('raw_equal_' + did)
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
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    p = json.loads(PROTOCOL.read_text())
    errors, modules = preflight(p)
    print(json.dumps({'status': 'preflight_failed' if errors else 'preflight_pass', 'errors': errors,
                      'execute_requested': args.execute, 'scientific_gate_credit': False}), flush=True)
    if errors:
        raise SystemExit(1)
    if args.execute:
        execute(p, modules)


if __name__ == '__main__':
    main()
