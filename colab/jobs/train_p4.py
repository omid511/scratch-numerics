"""Colab P4 training job: clone, fetch Drive data, train, write result file.

Usage: `colab run --gpu T4 --keep -s NAME colab/jobs/train_p4.py [branch]`.
Data: set DATA_FILE_ID to the Drive id of colab_data_upload.tgz.
Extracts to the repo root (p4_dataset/, data/p2/).
"""
import importlib.util
import subprocess
import sys

REPO = "https://github.com/omid511/scratch-numerics.git"
WORK = "/content/mech"
DATA_FILE_ID = "1KUgAyosjhSABPVhk1OrP15b0Gk41lMFy"
BRANCH = sys.argv[1] if len(sys.argv) > 1 else "master"

subprocess.run(
    f"rm -rf {WORK} && git clone --depth 1 -b {BRANCH} {REPO} {WORK}",
    shell=True, check=True, cwd="/content")
for pkg in ("gdown", "wandb"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run(f"pip install -q {pkg}", shell=True, check=True, cwd=WORK)
subprocess.run(
    f"python -c \"import gdown; gdown.download("
    f"'https://drive.google.com/uc?id={DATA_FILE_ID}', "
    f"'/content/data.tgz', quiet=False)\"",
    shell=True, check=True, cwd=WORK)
subprocess.run("tar -xzf /content/data.tgz", shell=True, check=True,
               cwd=WORK)
import os
env = dict(os.environ, WANDB_MODE="offline", WANDB_PROJECT="p4-margin")
with open("/content/result.txt", "w") as f:
    r = subprocess.run("PYTHONPATH=/content/mech/src python train_p4_expanded.py",
                       shell=True, cwd=WORK, stdout=f,
                       stderr=subprocess.STDOUT, env=env)
    f.write(f"[train exit={r.returncode}]\n")
artifacts = subprocess.run("ls -la /content/mech/*.pt /content/mech/*.json /content/mech/wandb 2>&1",
                           shell=True, check=False, cwd=WORK,
                           capture_output=True, text=True)
with open("/content/result.txt", "a") as f:
    f.write("[artifacts]\n" + artifacts.stdout + artifacts.stderr)
print("TRAIN_DONE — download /content/result.txt, *.pt, *.json and wandb/ before stopping",
      flush=True)
