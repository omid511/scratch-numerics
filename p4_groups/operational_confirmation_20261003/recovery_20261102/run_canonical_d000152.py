import hashlib, json, os, pathlib, resource, subprocess, sys, time, traceback

ROOT = pathlib.Path('/home/omid5/mechanics-recovery-20261102')
BASE = ROOT / 'p4_groups/operational_confirmation_20261003'
CONTROL = ROOT / 'recovery_control'
PROTOCOL = CONTROL / 'canonical_replay_protocol.json'
SCRIPT = BASE / 'experiment_support_modal_v2.py'
SOURCE = BASE / 'o1_corrected/trajectories_D000152.npz'
SUPPORT_PROTOCOL = BASE / 'support_modal_20261009_v2/protocol.json'
OUTPUT = BASE / 'support_modal_20261009_v2/shards/D000152.npz'
ACQUISITION_JSON = BASE / 'support_modal_20261009_v2/acquisition.json'
PRESERVED = CONTROL / 'preserved_mismatch_D000152_attempt1.npz'
PRESERVED_MAP = CONTROL / 'preserved_mismatch_D000152_attempt1_provenance.json'
PRIOR_RECEIPT = CONTROL / 'additional_attempt_receipt.json'
ADDITIONAL_PROTOCOL = CONTROL / 'additional_pilot_protocol.json'
SUPPLEMENTAL = CONTROL / 'supplemental_manifest_v5.json'
PROGRESS = ROOT / 'P4_PROGRESS.json'
RECEIPT = CONTROL / 'canonical_attempt_receipt.json'
LOG = CONTROL / 'canonical_attempt.log'
UNIT_ID = 'p4-recovery-d000152-canonical-20261102'
EXPECTED_SOURCE_BYTES = 3468164
EXPECTED_SOURCE_SHA256 = '699679e72bac7f78bce8e71715a69d6d601f1e23ebf1a75954a3b08c33f97033'
EXPECTED_OUTPUT_BYTES = 41270392
EXPECTED_OUTPUT_SHA256 = 'ca8594d4b5d13349a9d7e7edfd0218808884eecdd9687f1e593c607c2fdd3fdf'
PRIOR_BYTES = 41270304
PRIOR_SHA256 = 'd28379d0fd00d712e817cea7db4b846e4a5cc62f452ae184620064209450f7d2'
EXPECTED_PROGRESS_SHA256 = 'ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f'
EXPECTED_SUPPORT_PROTOCOL_SHA256 = '786f301fd82c64e9541365efb28c771b549f0272be13eaa16f4330801b4a4129'
EXPECTED_IMPLEMENTATION_SHA256 = '721a83d35bf68c9fb1011046e87b342d0e8170d4fd0f926e5ecabaf059ce5c6b'
EXPECTED_ADDITIONAL_PROTOCOL_SHA256 = '2f12dd0a034c12912d3b18261bcf356fe70a588bb8c2fcb6a404dd997c93ca5b'
EXPECTED_SUPPLEMENTAL_SHA256 = '4b1c9f2b3ddb5d7cb71db0c82b94dbd0c97dd8cb37dc3cc55fd738db0b7e9d17'
EXPECTED_PRIOR_RECEIPT_SHA256 = '568d75b928f9a69de72af171092386684bdbe32dea1555b62ab568df0fbf0b01'
EXPECTED_PRESERVED_MAP_SHA256 = 'b79c9e1a20dc6ad85cd6368bd8a59614cf7c08675665c9b2722103b226d23c50'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_proof(path):
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write_receipt(receipt):
    with RECEIPT.open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')


if RECEIPT.exists():
    raise FileExistsError(f'refusing to overwrite canonical evidence: {RECEIPT}')
if LOG.exists():
    raise FileExistsError(f'refusing to overwrite canonical log: {LOG}')

started = time.monotonic()
receipt = {
    'status': 'error',
    'unit_id': UNIT_ID,
    'total_timeout_seconds': 900,
    'acquisition_timeout_seconds': 600,
    'design': 'D000152',
    'prior_mismatch': {'bytes': PRIOR_BYTES, 'sha256': PRIOR_SHA256},
    'original_acceptance': {'bytes': EXPECTED_OUTPUT_BYTES, 'sha256': EXPECTED_OUTPUT_SHA256},
    'controller': {'python': sys.executable, 'python_version': sys.version, 'cwd': os.getcwd()},
}
log = LOG.open('x')
try:
    log.write('CANONICAL_CONTROLLER_START\n')
    log.write(json.dumps(receipt['controller'], sort_keys=True) + '\n')
    log.flush()
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['design'] != 'D000152' or protocol['prior_mismatch']['sha256'] != PRIOR_SHA256 or protocol['original_acceptance']['sha256'] != EXPECTED_OUTPUT_SHA256:
        raise ValueError('canonical replay protocol guard mismatch')
    for path, expected in (
        (SOURCE, EXPECTED_SOURCE_SHA256),
        (SCRIPT, EXPECTED_IMPLEMENTATION_SHA256),
        (SUPPORT_PROTOCOL, EXPECTED_SUPPORT_PROTOCOL_SHA256),
        (ADDITIONAL_PROTOCOL, EXPECTED_ADDITIONAL_PROTOCOL_SHA256),
        (SUPPLEMENTAL, EXPECTED_SUPPLEMENTAL_SHA256),
        (PRIOR_RECEIPT, EXPECTED_PRIOR_RECEIPT_SHA256),
        (PRESERVED_MAP, EXPECTED_PRESERVED_MAP_SHA256),
    ):
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'frozen guard mismatch: {path}')
    if SOURCE.stat().st_size != EXPECTED_SOURCE_BYTES:
        raise ValueError('D000152 context source size mismatch')
    if not PRESERVED.is_file() or PRESERVED.stat().st_size != PRIOR_BYTES or sha(PRESERVED) != PRIOR_SHA256:
        raise ValueError('preserved prior mismatch is missing or changed')
    if OUTPUT.exists():
        raise FileExistsError(f'canonical target is not empty: {OUTPUT}')
    if ACQUISITION_JSON.exists():
        raise FileExistsError(f'acquisition side-effect target already exists: {ACQUISITION_JSON}')
    if not PROGRESS.is_file() or sha(PROGRESS) != EXPECTED_PROGRESS_SHA256:
        raise ValueError('P4_PROGRESS changed')
    receipt['staged'] = {
        'source': file_proof(SOURCE),
        'implementation': file_proof(SCRIPT),
        'support_protocol': file_proof(SUPPORT_PROTOCOL),
        'additional_protocol': file_proof(ADDITIONAL_PROTOCOL),
        'supplemental_manifest': file_proof(SUPPLEMENTAL),
        'prior_receipt': file_proof(PRIOR_RECEIPT),
        'preserved_mismatch': file_proof(PRESERVED),
        'preserved_provenance': file_proof(PRESERVED_MAP),
        'progress': file_proof(PROGRESS),
        'acquisition_json_prelaunch': None,
    }
    command = [sys.executable, str(SCRIPT), '--phase', 'acquire', '--design', 'D000152']
    receipt['call'] = {'argv': command, 'cwd': str(ROOT), 'env': {key: os.environ.get(key) for key in ('PYTHONPATH', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG') if key in os.environ}}
    log.write('CLI_CALL ' + json.dumps(receipt['call'], sort_keys=True) + '\n')
    log.flush()
    call_started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
    receipt['call']['returncode'] = completed.returncode
    receipt['call']['elapsed_seconds'] = time.monotonic() - call_started
    log.flush()
    if OUTPUT.exists():
        receipt['output'] = file_proof(OUTPUT)
        receipt['output']['expected_bytes'] = EXPECTED_OUTPUT_BYTES
        receipt['output']['expected_sha256'] = EXPECTED_OUTPUT_SHA256
        if receipt['output']['bytes'] == EXPECTED_OUTPUT_BYTES and receipt['output']['sha256'] == EXPECTED_OUTPUT_SHA256 and completed.returncode == 0:
            receipt['status'] = 'pass'
        elif receipt['output']['bytes'] != EXPECTED_OUTPUT_BYTES or receipt['output']['sha256'] != EXPECTED_OUTPUT_SHA256:
            receipt['status'] = 'mismatch'
        else:
            receipt['status'] = 'error'
    elif completed.returncode != 0:
        receipt['status'] = 'error'
        receipt['error'] = {'type': 'CalledProcessError', 'message': f'CLI returned {completed.returncode} without output'}
    else:
        receipt['status'] = 'error'
        receipt['error'] = {'type': 'FileNotFoundError', 'message': 'CLI returned success without output shard'}
    if ACQUISITION_JSON.exists():
        receipt['acquisition_json'] = file_proof(ACQUISITION_JSON)
except BaseException as error:
    receipt['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    log.write('CONTROLLER_ERROR ' + receipt['error']['traceback'] + '\n')
finally:
    receipt['progress'] = {'sha256': sha(PROGRESS) if PROGRESS.exists() else None, 'expected_sha256': EXPECTED_PROGRESS_SHA256}
    if receipt['progress']['sha256'] != EXPECTED_PROGRESS_SHA256:
        receipt['status'] = 'error'
    receipt['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['total_seconds'] = time.monotonic() - started
    if receipt['total_seconds'] > 900:
        receipt['status'] = 'error'
        receipt['error'] = {'type': 'TimeoutError', 'message': 'total controller cap exceeded'}
    log.write('CANONICAL_CONTROLLER_END status=' + receipt['status'] + '\n')
    log.close()
    write_receipt(receipt)
    print('RECEIPT', RECEIPT, 'STATUS', receipt['status'], flush=True)
if receipt['status'] != 'pass':
    raise SystemExit(1)
