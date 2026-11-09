import hashlib, json, pathlib, resource, subprocess, sys, time, traceback

ROOT = pathlib.Path('/home/omid5/mechanics-recovery-20261102')
BASE = ROOT / 'p4_groups/operational_confirmation_20261003'
CONTROL = ROOT / 'recovery_control'
PROTOCOL_PATH = CONTROL / 'additional_pilot_protocol.json'
MANIFEST_PATH = CONTROL / 'supplemental_manifest_v5.json'
SOURCE = BASE / 'o1_corrected/trajectories_D000152.npz'
OUTPUT = BASE / 'support_modal_20261009_v2/shards/D000152.npz'
UNIT_ID = 'p4-recovery-d000152-20261102'
EXPECTED_SOURCE_BYTES = 3468164
EXPECTED_SOURCE_SHA256 = '699679e72bac7f78bce8e71715a69d6d601f1e23ebf1a75954a3b08c33f97033'
EXPECTED_OUTPUT_BYTES = 41270392
EXPECTED_OUTPUT_SHA256 = 'ca8594d4b5d13349a9d7e7edfd0218808884eecdd9687f1e593c607c2fdd3fdf'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_receipt(path, receipt):
    with path.open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')


def verify_manifest(manifest):
    for entry in manifest['files']:
        rel = pathlib.PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'unsafe staged path: {rel}')
        path = ROOT.joinpath(*rel.parts)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f'staged file mismatch: {rel}')


acquire = len(sys.argv) == 2 and sys.argv[1] == '--acquire'
preflight = not sys.argv[1:]
if not (acquire or preflight):
    raise SystemExit('usage: run_pilot_d000152.py [--acquire]')
receipt_path = CONTROL / ('additional_attempt_receipt.json' if acquire else 'd000152_preflight_receipt.json')
if receipt_path.exists():
    raise FileExistsError(f'refusing to overwrite evidence: {receipt_path}')
started = time.monotonic()
receipt = {
    'status': 'error',
    'mode': 'acquire' if acquire else 'preflight',
    'unit_id': UNIT_ID,
    'total_timeout_seconds': 900,
    'acquisition_timeout_seconds': 600,
    'acquisition_scope': 'Independent subprocess deadline includes child imports, contract validation and acquire_design' if acquire else 'No acquisition; readonly contract, inputs, O1 context load and plate validation only',
    'design': 'D000152',
    'source': {'path': str(SOURCE.relative_to(ROOT)), 'bytes': EXPECTED_SOURCE_BYTES, 'sha256': EXPECTED_SOURCE_SHA256},
    'expected_output': {'path': str(OUTPUT.relative_to(ROOT)), 'bytes': EXPECTED_OUTPUT_BYTES, 'sha256': EXPECTED_OUTPUT_SHA256},
}
try:
    if not SOURCE.is_file() or SOURCE.stat().st_size != EXPECTED_SOURCE_BYTES or sha(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError('D000152 context source size/hash mismatch')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol['design'] != 'D000152' or protocol['context_source']['bytes'] != EXPECTED_SOURCE_BYTES or protocol['context_source']['sha256'] != EXPECTED_SOURCE_SHA256:
        raise ValueError('additional pilot protocol context source mismatch')
    if protocol['expected_shard']['bytes'] != EXPECTED_OUTPUT_BYTES or protocol['expected_shard']['sha256'] != EXPECTED_OUTPUT_SHA256:
        raise ValueError('additional pilot protocol expected output mismatch')
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_entry = next((entry for entry in manifest['files'] if entry['path'] == str(SOURCE.relative_to(ROOT))), None)
    if manifest_entry != {'path': str(SOURCE.relative_to(ROOT)), 'bytes': EXPECTED_SOURCE_BYTES, 'sha256': EXPECTED_SOURCE_SHA256}:
        raise ValueError('supplemental v5 context entry mismatch')
    verify_manifest(manifest)
    receipt['supplemental_manifest'] = {'path': str(MANIFEST_PATH.relative_to(ROOT)), 'entries': len(manifest['files']), 'sha256': sha(MANIFEST_PATH)}
    import importlib
    importlib.invalidate_caches()
    import_started = time.monotonic()
    import numpy, scipy, torch
    receipt['runtime'] = {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    import experiment_support_modal_v2 as experiment
    receipt['import_seconds'] = time.monotonic() - import_started
    contract_started = time.monotonic()
    p = experiment.contract()
    d, fp = experiment.inputs()
    identity, plate, audit = experiment._load_o1_context(d, fp, 'D000152')
    validation = experiment.validate_plate(plate, p, audit)
    receipt['contract_and_readonly_preflight_seconds'] = time.monotonic() - contract_started
    receipt['preflight'] = {
        'design': plate.design.design_id,
        'profile_count': len(p['applicable_profiles']['D000152']),
        'profiles': list(p['applicable_profiles']['D000152']),
        'plate_count': plate.count,
        'minimum_mach': validation['minimum_mach'],
        'extreme_count': len(validation['extreme']),
        'fingerprint_sha256': hashlib.sha256(json.dumps(fp, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'acquisition_supported': validation['acquisition_supported'],
    }
    if preflight:
        receipt['status'] = 'preflight_pass'
    else:
        if OUTPUT.exists():
            raise FileExistsError(f'refusing to reuse existing shard: {OUTPUT}')
        acquisition_started = time.monotonic()
        subprocess.run([sys.executable, '-c', "import experiment_support_modal_v2 as e; p=e.contract(); d, fp=e.inputs(); e.acquire_design('D000152', p, d, fp)"], cwd=ROOT, check=True, timeout=600)
        receipt['acquisition_seconds'] = time.monotonic() - acquisition_started
        receipt['output'] = {'path': str(OUTPUT.relative_to(ROOT)), 'bytes': OUTPUT.stat().st_size, 'sha256': sha(OUTPUT), 'expected_bytes': EXPECTED_OUTPUT_BYTES, 'expected_sha256': EXPECTED_OUTPUT_SHA256}
        receipt['status'] = 'pass' if receipt['output']['bytes'] == EXPECTED_OUTPUT_BYTES and receipt['output']['sha256'] == EXPECTED_OUTPUT_SHA256 else 'mismatch'
except BaseException as error:
    receipt['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
finally:
    progress = ROOT / 'P4_PROGRESS.json'
    receipt['progress'] = {'sha256': sha(progress) if progress.exists() else None, 'expected_sha256': 'ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f'}
    if receipt['progress']['sha256'] != receipt['progress']['expected_sha256']:
        receipt['status'] = 'error'
    receipt['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['total_seconds'] = time.monotonic() - started
    if receipt['total_seconds'] > 900:
        receipt['status'] = 'error'
        receipt['error'] = {'type': 'TimeoutError', 'message': 'total controller cap exceeded'}
    write_receipt(receipt_path, receipt)
    print('RECEIPT', receipt_path, 'STATUS', receipt['status'], flush=True)
    if 'error' in receipt:
        print('ERROR', receipt['error']['traceback'], flush=True)
if receipt['status'] not in ('pass', 'preflight_pass'):
    raise SystemExit(1)
