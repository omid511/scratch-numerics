"""VM-side P4 training launcher: clone branch, stage data, train under nohup.
Usage: colab exec -s p4train -f colab/jobs/launch_p4_train.py --timeout 600
Poll:  colab exec -s p4train -f colab/jobs/poll_p4_train.py
Dataset: /content/p4data.tgz chunks/upload, else the p4-data-v2 release asset.
"""
import subprocess
import sys
REPO = "https://github.com/omid511/scratch-numerics.git"
WORK = "/content/mech"
BRANCH = sys.argv[1] if len(sys.argv) > 1 else "p4-review-fixes"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    kw.setdefault("cwd", WORK)
    r = subprocess.run(cmd, shell=True, **kw)
    print(f"[exit={r.returncode}]", flush=True)
    return r

sh(f"rm -rf {WORK} && git clone --depth 1 -b {BRANCH} {REPO} {WORK}", cwd="/content", check=True)
sh("git rev-parse --short HEAD", check=True)
sh("mkdir -p p4_dataset", check=True)
r = subprocess.run("ls /content/p4data.tgz.aa 2>/dev/null && echo HAVE_CHUNKS; "
                   "test -f /content/p4data.tgz && echo HAVE_UPLOAD", shell=True, cwd=WORK,
                   capture_output=True, text=True)
if "HAVE_CHUNKS" in r.stdout:
    sh("cat /content/p4data.tgz.* > /content/p4data.tgz && tar -xzf /content/p4data.tgz -C p4_dataset", check=True)
elif "HAVE_UPLOAD" in r.stdout:
    sh("tar -xzf /content/p4data.tgz -C p4_dataset", check=True)
else:
    # No Drive fallback: the old Drive corpus lacks provenance and must never
    # silently substitute for the regenerated dataset.
    print("no upload found; pulling release asset (VM egress is reliable)", flush=True)
    sh("curl -sSL -o /content/p4data.tgz "
       "https://github.com/omid511/scratch-numerics/releases/download/p4-data-v2/p4data.tgz "
       "&& tar -xzf /content/p4data.tgz -C p4_dataset", check=True)
sh("pip install -q wandb", check=True)
sh("nvidia-smi -L", check=False)
# Launch training detached; poll result.txt separately. WANDB offline: sync later.
sh("WANDB_MODE=offline WANDB_PROJECT=p4-margin PYTHONPATH=/content/mech/src "
   "nohup python train_p4_expanded.py > /content/train.log 2>&1 & echo started pid=$!", check=True)
print("LAUNCHED — poll with poll_p4_train.py", flush=True)
