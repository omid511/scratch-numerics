import hashlib, importlib, json, pathlib, resource, subprocess, sys, tarfile, time, traceback
root = pathlib.Path('/home/omid5/mechanics-recovery-20261102')
base = root / 'p4_groups/operational_confirmation_20261003'
control = root / 'recovery_control'
receipt_path = control / 'attempt4_receipt.json'
archive = root / 'pilot.tar.gz'
expected_archive = '717bcf5e241c8a40c0b658a277812a679922d0c255bd9b2a45083882cc7aa23a'
expected_progress = 'ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f'
manifest_path = base / 'recovery_20261102/input_manifest.json'
supplement_path = control / 'supplemental_manifest_v4.json'
output = base / 'support_modal_20261009_v2/shards/D000007.npz'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
if receipt_path.exists():
    raise FileExistsError(f'refusing to overwrite attempt evidence: {receipt_path}')
receipt = {'status': 'error', 'unit_id': 'p4-recovery-d000007-attempt4-20261102', 'total_timeout_seconds': 900, 'transfer_seconds': float(sys.argv[1]), 'transfer_scope': 'Successful suffix-resume attempt only; preceding cancelled transfer duration is unknown', 'python': sys.version, 'acquisition_timeout_seconds': 600, 'acquisition_scope': 'Independent subprocess deadline includes child imports, contract validation and acquire_design'}
started = time.monotonic()
try:
    if output.exists(): raise FileExistsError(f'refusing to reuse existing shard: {output}')
    receipt['archive'] = {'bytes': archive.stat().st_size, 'sha256': sha(archive), 'expected_sha256': expected_archive}
    if receipt['archive']['bytes'] != 95507253 or receipt['archive']['sha256'] != expected_archive: raise ValueError('archive mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        member = tar.getmember('p4_groups/operational_confirmation_20261003/recovery_20261102/input_manifest.json')
        with tar.extractfile(member) as f: trusted_manifest = f.read()
    if manifest_path.read_bytes() != trusted_manifest: raise ValueError('staged manifest differs from trusted archive')
    manifest = json.loads(trusted_manifest)
    supplemental = json.loads(supplement_path.read_text())
    receipt['input_manifest'] = manifest
    receipt['supplemental_manifest'] = supplemental
    receipt['supplemental_manifest_sha256'] = sha(supplement_path)
    for entry in manifest['files'] + supplemental['files']:
        rel = pathlib.PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError(f'unsafe staged path: {rel}')
        file = root.joinpath(*rel.parts)
        if file.is_symlink() or not file.is_file() or file.stat().st_size != entry['bytes'] or sha(file) != entry['sha256']: raise ValueError(f'staged file mismatch: {rel}')
    receipt['staged_verification_seconds'] = time.monotonic() - started
    importlib.invalidate_caches()
    import_started = time.monotonic()
    import numpy, scipy, torch
    receipt['runtime'] = {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    import experiment_support_modal_v2 as experiment
    receipt['import_seconds'] = time.monotonic() - import_started
    print('IMPORT_OK', json.dumps(receipt['runtime']), 'seconds', receipt['import_seconds'], flush=True)
    contract_started = time.monotonic()
    p = experiment.contract()
    d, fp = experiment.inputs()
    receipt['contract_and_inputs_seconds'] = time.monotonic() - contract_started
    acquire_started = time.monotonic()
    try:
        subprocess.run([sys.executable, '-c', "import experiment_support_modal_v2 as e; p=e.contract(); d, fp=e.inputs(); e.acquire_design('D000007', p, d, fp)"], cwd=root, check=True, timeout=600)
    finally:
        receipt['acquisition_seconds'] = time.monotonic() - acquire_started
    receipt['output'] = {'path': str(output.relative_to(root)), 'bytes': output.stat().st_size, 'sha256': sha(output), 'expected_bytes': 42258789, 'expected_sha256': 'cd203cee952cba0b64365780b0da3e47e632664b38fda8628ae1f4a8539921d7'}
    receipt['status'] = 'pass' if receipt['output']['bytes'] == receipt['output']['expected_bytes'] and receipt['output']['sha256'] == receipt['output']['expected_sha256'] else 'mismatch'
except BaseException as error:
    receipt['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
finally:
    progress = root / 'P4_PROGRESS.json'
    receipt['progress'] = {'sha256': sha(progress) if progress.exists() else None, 'expected_sha256': expected_progress}
    if receipt['progress']['sha256'] != expected_progress: receipt['status'] = 'error'
    receipt['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['total_seconds'] = time.monotonic() - started
    with receipt_path.open('x') as f: json.dump(receipt, f, indent=2, allow_nan=False); f.write('\n')
    print('RECEIPT', receipt_path, 'STATUS', receipt['status'], flush=True)
    if 'error' in receipt: print('ERROR', receipt['error']['traceback'], flush=True)
if receipt['status'] != 'pass': sys.exit(1)
