"""Model-level checkpoint helpers for preemption-resilient training runs.

Checkpoints are plain ``torch.save`` dicts ``{"epoch", "state", "history"}``
under a run directory (default ``ckpts/``, gitignored via ``*.ckpt``).
Resume granularity is one model: reruns skip models whose final weights
exist; epoch checkpoints exist for forensics and future warm restarts
(current trainers always train from scratch — worst case per preemption
is one model, ~21 T4-minutes).
"""
from __future__ import annotations

import glob
import os

import torch


def ckpt_path(ckpt_dir: str, name: str, epoch: int) -> str:
    return os.path.join(ckpt_dir, f"{name}_e{epoch:03d}.ckpt")


def save_ckpt(path: str, model: torch.nn.Module, epoch: int, history: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "history": {k: list(v) for k, v in history.items()},
    }, path)


def latest_ckpt(ckpt_dir: str, name: str) -> str | None:
    cands = sorted(glob.glob(os.path.join(ckpt_dir, f"{name}_e*.ckpt")))
    return cands[-1] if cands else None
