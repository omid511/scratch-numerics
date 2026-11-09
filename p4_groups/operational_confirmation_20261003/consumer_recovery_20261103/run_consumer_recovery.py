"""One separately recorded native reconstruction; original consumer hashes are the gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
PROTOCOL = HERE / 'protocol_v2.json'
RUN = HERE / 'run_001'
PROGRESS = ROOT / 'P4_PROGRESS.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def checked_protocol():
    p = json.loads(PROTOCOL.read_text())
    if sha(__file__) != p['controller_sha256']:
        raise RuntimeError('controller fingerprint changed')
    if sha(HERE / 'required_input_manifest.json') != p['input_manifest_sha256']:
        raise RuntimeError('input manifest fingerprint changed')
    if sha(PROGRESS) != p['progress_sha256']:
        raise RuntimeError('protected progress fingerprint changed')
    return p


def check_file(path, proof):
    if path.stat().st_size != proof['bytes'] or sha(path) != proof['sha256']:
        raise RuntimeError(f'input fingerprint changed: {path}')


def native_modules():
    sys.path[:0] = [str(ROOT / 'src'), str(BASE)]
    import raw_modal_data as D
    import experiment_support_modal_v2 as S
    if D.E is not S:
        raise RuntimeError('normalization and generation do not share native support module')
    S.SHARDS = RUN / 'shards'
    D.OUT = RUN / 'artifact'
    return D, S


def child(p, design, normalize):
    from threadpoolctl import threadpool_limits
    D, S = native_modules()
    with threadpool_limits(limits=1):
        native_p = S.contract()
        if normalize:
            D.acquire()
            wave, valid = D.load_measured()
            import numpy as np
            if wave.shape != (2094, 16, 8001) or wave.dtype != np.float32:
                raise RuntimeError('waveform shape/dtype changed')
            if valid.shape != (2094, 8001) or valid.dtype != np.bool_:
                raise RuntimeError('admissibility shape/dtype changed')
            if wave.flags.writeable or valid.flags.writeable:
                raise RuntimeError('native consumer load is not read-only')
        else:
            if design not in p['designs']:
                raise RuntimeError('design outside frozen consumer manifest')
            d, fp = S.inputs()
            S.acquire_design(design, native_p, d, fp)
    checked_protocol()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--design')
    mode.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    p = checked_protocol()
    if args.design or args.normalize:
        child(p, args.design, args.normalize)
        return
    manifest = json.loads((HERE / 'required_input_manifest.json').read_text())
    for name, proof in manifest['files'].items():
        check_file(ROOT / name, proof)
    for did, proof in p['reuse_shards'].items():
        check_file(BASE / 'support_modal_20261009_v2/shards' / f'{did}.npz', proof)
    if RUN.exists():
        raise FileExistsError(f'refusing existing run namespace: {RUN}')
    if not args.execute:
        print(json.dumps({'status': 'inputs_verified', 'designs': len(p['designs']),
                          'generation_exercised': False, 'consumer_identity_verified': False}))
        return
    RUN.mkdir()
    (RUN / 'shards').mkdir()
    started = time.monotonic()
    receipt = {'status': 'error', 'protocol_sha256': sha(PROTOCOL),
               'input_manifest_sha256': p['input_manifest_sha256'], 'shards': {},
               'consumer_hashes': {}, 'scientific_gate_credit': False}
    try:
        for did, proof in p['reuse_shards'].items():
            target = RUN / 'shards' / f'{did}.npz'
            shutil.copyfile(BASE / 'support_modal_20261009_v2/shards' / target.name, target)
            check_file(target, proof)
        env = dict(os.environ, **p['thread_environment'])
        stages = [(did, ['--design', did], 120) for did in p['designs']]
        stages.append(('normalize', ['--normalize'], 180))
        for name, flags, stage_cap in stages:
            remaining = p['total_timeout_seconds'] - (time.monotonic() - started) - 10
            if remaining <= 0:
                raise TimeoutError('total reconstruction deadline reached')
            argv = [sys.executable, str(Path(__file__).resolve()), *flags]
            save(RUN / 'checkpoint.json', {'stage': name, 'argv': argv,
                 'elapsed_seconds': time.monotonic() - started,
                 'execution_owner': p['execution_owner'], 'resume': 'inspect receipt/unit; never rerun existing run_001'})
            stage_started = time.monotonic()
            with (RUN / f'{name}.log').open('x') as log:
                subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=min(stage_cap, remaining))
            checked_protocol()
            if name != 'normalize':
                path = RUN / 'shards' / f'{name}.npz'
                receipt['shards'][name] = {'path': str(path.relative_to(ROOT)),
                    'bytes': path.stat().st_size, 'sha256': sha(path),
                    'stage_seconds': time.monotonic() - stage_started,
                    'original_reference': p['original_shard_references'][name]}
                save(RUN / 'source_manifest.json', receipt['shards'])
        for filename, expected in p['consumer_hashes'].items():
            path = RUN / 'artifact' / filename
            actual = sha(path)
            receipt['consumer_hashes'][filename] = {'bytes': path.stat().st_size,
                'sha256': actual, 'expected_sha256': expected, 'match': actual == expected}
        receipt['acquisition_metadata_sha256'] = sha(RUN / 'artifact/acquisition.json')
        receipt['status'] = 'exact_consumer_hashes' if all(
            row['match'] for row in receipt['consumer_hashes'].values()) else 'consumer_hash_mismatch'
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error),
                            'traceback': traceback.format_exc()}
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['progress_sha256'] = sha(PROGRESS)
        if receipt['progress_sha256'] != p['progress_sha256']:
            receipt['status'] = 'error'
        save(RUN / 'receipt.json', receipt, exclusive=True)
        print(json.dumps({'status': receipt['status'], 'receipt': str(RUN / 'receipt.json')}))
    if receipt['status'] != 'exact_consumer_hashes':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
