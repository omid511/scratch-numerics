"""Pack P4 training artifacts into one tarball for download."""
import subprocess

WORK = "/content/mech"


def sh(cmd):
    r = subprocess.run(cmd, shell=True, cwd=WORK, capture_output=True, text=True)
    print(r.stdout.strip())
    print(r.stderr.strip())


sh("ls -la /content/mech/*.pt /content/mech/p4_train_results.json 2>&1 | tail -20")
sh("tar -czf /content/p4artifacts.tgz -C /content/mech "
   "$(ls /content/mech/p4_*.pt 2>/dev/null | xargs -n1 basename | tr '\n' ' ') "
   "p4_train_results.json wandb 2>&1 && ls -la /content/p4artifacts.tgz")
print("PACK_DONE")
