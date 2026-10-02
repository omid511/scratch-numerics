"""Sample-weighted optimization and best-validation restoration for P4 models."""
from __future__ import annotations

import math
import torch


def fit_epochs(model, train_dl, val_dl, loss_for_batch, *, epochs, lr,
               validation_loss=None, on_epoch=None, patience=None, weight_decay=0.0,
               schedule_epochs=None):
    """Fit with Adam, cosine decay and norm-1 gradient clipping.

    Loss callbacks receive ``(model, batch)`` and return a scalar mean loss.
    The first batch tensor defines the sample count (pairs for paired training).
    Validation is sample-weighted, never an unweighted mean of minibatch means.
    ``on_epoch`` observes completed epochs before the best state is restored.
    ``val_dl=None`` performs a fixed-epoch refit without model selection.
    """
    if epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive")
    validation_loss = validation_loss or loss_for_batch
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule_epochs or epochs)
    history = {"train_loss": [], "val_loss": []}
    best, best_state, stale = math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        total = torch.zeros((), device=device)
        count = 0
        for batch in train_dl:
            batch = tuple(value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            n = len(batch[0])
            total += loss.detach() * n
            count += n
        if not count:
            raise ValueError("Empty training loader")
        history["train_loss"].append(float(total / count))
        scheduler.step()
        model.eval()
        if val_dl is None:
            if on_epoch is not None:
                on_epoch(epoch, model, history)
            continue
        total = torch.zeros((), device=device)
        count = 0
        with torch.inference_mode():
            for batch in val_dl:
                batch = tuple(value.to(device) for value in batch)
                n = len(batch[0])
                total += validation_loss(model, batch) * n
                count += n
        if not count:
            raise ValueError("Empty validation loader")
        value = float(total / count)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss")
        history["val_loss"].append(value)
        if value < best:
            best = value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(epoch, model, history)
        if patience is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
