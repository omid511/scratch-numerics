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
    args = parser.parse_args()
    label = 'smoke' if args.smoke else 'full'
    status_path = OUT / f'{label}_status.json'
    command = [sys.executable, '-u', str(OUT / 'experiment_temporal.py'), '--workers', str(args.workers)]
    if args.smoke:
        command.append('--smoke')
    record = dict(phase='running', command=command, started=timestamp(), supervisor_pid=os.getpid(),
                  log=str(OUT / f'{label}_run.log'), exit_code=None)
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_path)
    with (OUT / f'{label}_run.log').open('a', buffering=1) as log:
        log.write(f'\nSTART {record["started"]} {command}\n')
        child = subprocess.Popen(command, cwd=OUT.parents[1], stdout=log, stderr=subprocess.STDOUT)
        record['child_pid'] = child.pid
        save()
        def terminate(signum, frame):
            record['termination_signal'] = signum
            child.terminate()
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        print(f'TEMPORAL_SUPERVISOR_READY {label} pid={child.pid}', flush=True)
        code = child.wait()
        record.update(phase='complete' if code == 0 else 'failed', exit_code=code, finished=timestamp())
        save()
        log.write(f'EXIT {record["finished"]} {code}\n')
    print(f'TEMPORAL_SUPERVISOR_EXIT {label} code={code}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
