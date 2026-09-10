"""Poll P4 training: log tail + process state. Exit 0 always (informational)."""
import subprocess

WORK = "/content/mech"


def sh(cmd):
    r = subprocess.run(cmd, shell=True, cwd=WORK, capture_output=True, text=True)
    return r.stdout.strip() + r.stderr.strip()


print("--- train.log tail ---")
print(sh("tail -c 3000 /content/train.log"))
print("--- trainer alive ---")
print(sh("pgrep -af train_p4_expanded || echo NO_TRAINER_PROC"))
print("--- artifacts ---")
print(sh("ls -la /content/mech/*.pt /content/mech/p4_train_results.json /content/mech/wandb 2>&1 | tail -25"))
