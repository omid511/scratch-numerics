import hashlib, json, os, pathlib, resource, subprocess, sys, tarfile, time, traceback
root = pathlib.Path('/home/omid5/mechanics-recovery-20261102')
base = root / 'p4_groups/operational_confirmation_20261003'
recovery = base / 'recovery_20261102'
recovery.mkdir(parents=True, exist_ok=True)
receipt_path = recovery / 'execution_receipt.json'
archive = root / 'pilot.tar.gz'
expected_archive = '717bcf5e241c8a40c0b658a277812a679922d0c255bd9b2a45083882cc7aa23a'
expected_progress = 'ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
receipt = {'status': 'error', 'archive': {'bytes': archive.stat().st_size, 'sha256': sha(archive), 'expected_sha256': expected_archive}, 'transfer_seconds': float(sys.argv[1]), 'python': sys.version, 'call': "contract(); inputs(); acquire_design('D000007', p, d, fp)", 'acquisition_timeout_seconds': 600}
receipt['transfer_scope'] = 'Successful suffix-resume attempt only; preceding cancelled transfer duration is unknown'
receipt['acquisition_scope'] = 'Independent subprocess deadline includes child imports, contract validation and acquire_design'
started = time.monotonic()
try:
    if receipt['archive']['sha256'] != expected_archive: raise ValueError('archive SHA256 mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile(): raise ValueError(f'unsafe archive member: {member.name}')
            destination = root.joinpath(*path.parts)
            if destination.exists() or destination.is_symlink(): raise ValueError(f'archive destination already exists: {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open('xb') as target:
                while data := source.read(1024 * 1024): target.write(data)
    manifest = json.loads((recovery / 'input_manifest.json').read_text())
    receipt['input_manifest'] = manifest
    for entry in manifest['files']:
        file = root / entry['path']
        if file.stat().st_size != entry['bytes'] or sha(file) != entry['sha256']: raise ValueError(f"staged file mismatch: {entry['path']}")
    receipt['extraction_verification_seconds'] = time.monotonic() - started
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
        subprocess.run(
            [sys.executable, '-c', "import experiment_support_modal_v2 as e; p=e.contract(); d, fp=e.inputs(); e.acquire_design('D000007', p, d, fp)"],
            cwd=root, check=True, timeout=600,
        )
        output = base / 'support_modal_20261009_v2/shards/D000007.npz'
    finally:
        receipt['acquisition_seconds'] = time.monotonic() - acquire_started
    output = pathlib.Path(output)
    expected_path = base / 'support_modal_20261009_v2/shards/D000007.npz'
    if output != expected_path: raise ValueError(f'unexpected output path: {output}')
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
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print('RECEIPT', receipt_path, 'STATUS', receipt['status'], flush=True)
    if 'output' in receipt: print('OUTPUT', json.dumps(receipt['output']), flush=True)
    if 'error' in receipt: print('ERROR', receipt['error']['traceback'], flush=True)
if receipt['status'] != 'pass': sys.exit(1)
