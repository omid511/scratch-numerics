# Colab harness — rerunnable tests & training

All jobs are plain Python scripts executed with `colab exec -s NAME -f` or
`colab run [--gpu T4]`. Nothing here needs a TTY. VMs are ephemeral: every
run re-clones, re-fetches data, writes `/content/result.txt`, and the
operator downloads it before `colab stop`.

## Data (Google Drive, anyone-with-link)

| File (Drive) | Contents | Laid out at |
|---|---|---|
| `colab_data_upload.tgz` (~94MB) | `p4_dataset/` (6122 clips), `data/p2/` | repo root |

Fetch on VM: `pip install gdown && python -c "import gdown;
gdown.download('https://drive.google.com/uc?id=<ID>', '/content/data.tgz')"`.
P1 COMSOL shapes are VOID (see P1_PART1_MVP_RESULTS.md) — do not stage them.

## Runbook

```bash
colab new -s NAME [--gpu T4]          # or: colab run --keep -s NAME [--gpu T4] job.py
colab exec -s NAME -f colab/jobs/<job>.py --timeout 3300
colab download -s NAME /content/result.txt ./result.txt
colab stop -s NAME                    # ALWAYS — idle VMs burn units, then get reaped with your artifacts
colab sessions                        # census before AND after every provision
```

## Fuck-up rules (earned)

1. Census before/after every provision. A `[?]` orphan counts against quota
   (even `TooManyAssignmentsError` on CPU) — release it, don't just report it.
2. Prefer `colab run` (self-cleans); named sessions (`-s`) only when you must download.
3. Never blind `pip install` — pristine images already carry numpy/scipy/torch;
   install only what's missing or the resolver can wedge the kernel for 20+ min.
4. `PYTHONPATH=<root>/src` — the package is src-layout, not installed.
5. Root `tests/` exclude `test_cross_validation.py` (needs legacy `plate` package).
6. Pytest output streaming flakes: jobs write `/content/result.txt`, fetch via `download`.
7. Download artifacts (weights, results) BEFORE stopping — reaped VMs take everything.
8. scipy eig is CPU-bound: slow tests want longer caps, not a GPU. T4 pays off for torch training only.
9. `colab exec`/`run` drop ~50% of provisioning RPCs (`RemoteDisconnected`); retry
   with census, never stack attempts blindly.
10. `colab stop` cannot kill `[?]` orphans (no local record) and the web UI is
    not the only way — release the endpoint directly (no stray daemons needed;
    verify with `ps` + empty `sessions.json` first):

```bash
colab sessions   # copy the full endpoint after [?], e.g. gpu-t4-s-kkb-...-27mgvc4x7vfxe
<tool-python> -c "from colab_cli.common import state; \
  state.client.unassign('<endpoint>')"
colab sessions   # expect: No active sessions found on server.
```

`<tool-python>` is the CLI's own interpreter, e.g.
`~/.local/share/uv/tools/google-colab-cli/bin/python` (must share the default
`~/.config/colab-cli/sessions.json` state). Then retry the provision.
