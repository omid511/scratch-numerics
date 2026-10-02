"""Run the fixed experiment with durable logs and an explicit exit receipt."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

OUT = Path(__file__).resolve().parent


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pipeline', action='store_true')
    args = parser.parse_args()
    if args.pipeline and args.smoke:
        parser.error('--pipeline includes its own smoke stage')
    label = 'pipeline' if args.pipeline else ('smoke' if args.smoke else 'full')
    status_path = OUT / f'{label}_status.json'
    command = [sys.executable, '-u', str(OUT / 'experiment_temporal.py'), '--workers', str(args.workers)]
    if args.smoke:
        command.append('--smoke')
    stages = [('training', command)]
    if args.pipeline:
        def invocation(script, *flags):
            return [sys.executable, '-u', str(OUT / script), *flags]
        stages = [
            ('prepare', invocation('experiment_temporal.py', '--prepare')),
            ('smoke', command + ['--smoke']),
            ('smoke_scoring', invocation('evaluate_temporal.py', '--smoke')),
            ('smoke_verification', invocation('verify_temporal.py', '--smoke')),
            ('full_training', command),
            ('full_scoring', invocation('evaluate_temporal.py')),
            ('full_verification', invocation('verify_temporal.py')),
        ]
    record = dict(phase='running', command=command, started=timestamp(), supervisor_pid=os.getpid(),
                  log=str(OUT / f'{label}_run.log'), exit_code=None)
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_path)
    with (OUT / f'{label}_run.log').open('a', buffering=1) as log:
        child = None
        def terminate(signum, frame):
            record['termination_signal'] = signum
            if child is not None:
                child.terminate()
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        code = 0
        for stage, command in stages:
            if 'termination_signal' in record:
                code = 128 + record['termination_signal']
                break
            record.update(stage=stage, command=command)
            log.write(f'\nSTART {timestamp()} {stage} {command}\n')
            child = subprocess.Popen(command, cwd=OUT.parents[1], stdout=log, stderr=subprocess.STDOUT)
            record['child_pid'] = child.pid
            save()
            print(f'TEMPORAL_SUPERVISOR_READY {label} stage={stage} pid={child.pid}', flush=True)
            code = child.wait()
            log.write(f'EXIT {timestamp()} {stage} {code}\n')
            if code != 0:
                break
        record.update(phase='complete' if code == 0 else 'failed', exit_code=code, finished=timestamp())
        save()
    print(f'TEMPORAL_SUPERVISOR_EXIT {label} code={code}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
