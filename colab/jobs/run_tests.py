"""Colab test job: clone, add only missing deps, run pytest, write result file.

Usage (on VM via `colab exec -s NAME -f colab/jobs/run_tests.py`):
    python colab/jobs/run_tests.py [branch] [pytest-target...]
Defaults: branch=master, target="src tests --ignore=tests/test_cross_validation.py".
"""
import importlib.util
import subprocess
import sys

REPO = "https://github.com/omid511/scratch-numerics.git"
WORK = "/content/mech"
BRANCH = sys.argv[1] if len(sys.argv) > 1 else "master"
TARGET = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else \
    "src tests --ignore=tests/test_cross_validation.py"

subprocess.run(
    f"rm -rf {WORK} && git clone --depth 1 -b {BRANCH} {REPO} {WORK}",
    shell=True, check=True, cwd="/content")
print("HEAD:", subprocess.run("git rev-parse --short HEAD", shell=True,
                              cwd=WORK, capture_output=True,
                              text=True).stdout.strip(), flush=True)
for mod in ["pytest", "pytest_timeout"]:
    if importlib.util.find_spec(mod) is None:
        subprocess.run(f"pip install -q {mod.replace('_', '-')}",
                       shell=True, check=True, cwd=WORK)
with open("/content/result.txt", "w") as f:
    r = subprocess.run(
        f"PYTHONPATH={WORK}/src python -m pytest {TARGET} -q "
        f"-p no:cacheprovider --timeout=900",
        shell=True, cwd=WORK, stdout=f, stderr=subprocess.STDOUT)
    f.write(f"[exit={r.returncode}]\n")
print("TESTS_DONE", flush=True)
