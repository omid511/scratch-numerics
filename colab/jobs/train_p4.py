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
if importlib.util.find_spec("gdown") is None:
    subprocess.run("pip install -q gdown", shell=True, check=True, cwd=WORK)
subprocess.run(
    f"python -c \"import gdown; gdown.download("
    f"'https://drive.google.com/uc?id={DATA_FILE_ID}', "
    f"'/content/data.tgz', quiet=False)\"",
    shell=True, check=True, cwd=WORK)
subprocess.run("tar -xzf /content/data.tgz", shell=True, check=True,
               cwd=WORK)
with open("/content/result.txt", "w") as f:
    r = subprocess.run("PYTHONPATH=/content/mech/src python train_p4_expanded.py",
                       shell=True, cwd=WORK, stdout=f,
                       stderr=subprocess.STDOUT)
    f.write(f"[train exit={r.returncode}]\n")
print("TRAIN_DONE — download /content/result.txt and any *.pt before stopping",
      flush=True)
