"""Anchor-vs-reversal controls layered on the frozen fresh-study model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fresh.models import make_model


class _CompactBase(nn.Module):
    """Reuse exactly the Compact tokenizer, forward TCN, and classifier."""

    def __init__(self, base):
        super().__init__()
        self.tokenizer = base.tokenizer
        self.forward_reader = base.forward_reader
        self.classifier = base.classifier
        self.width = base.width

    @torch.no_grad()
    def project_constraints(self):
        if hasattr(self.tokenizer, "project_constraints"):
            self.tokenizer.project_constraints()
        if hasattr(self.classifier, "project_constraints"):
            self.classifier.project_constraints()


class FixedAnchorForward(_CompactBase):
    """Chronological processing with one fixed terminal summary token.

    The classifier always reads the TCN state corresponding to the same learned
    anchor token, independent of the number of observed EEG tokens.  The EEG
    itself is never reversed.

    The anchor adds only `width` trainable parameters (64 in the paper).
    """

    def __init__(self, base):
        super().__init__(base)

        # Initialize from the endpoint positional vector without consuming a
        # second random stream. It is then an independent learned summary token.
        self.anchor_token = nn.Parameter(
            self.tokenizer.position[:, :, -1:].detach().clone()
        )

    def forward(self, x):
        z = self.tokenizer(x)
        anchor = self.anchor_token.expand(z.shape[0], -1, -1)
        z_anchor = torch.cat([z, anchor], dim=-1)

        h = self.forward_reader(z_anchor)[:, :, -1]
        return self.classifier(h)


class RelativePositionForward(_CompactBase):
    """Latest-token forward readout with positions aligned to the endpoint.

    For an n-token prefix, use the final n vectors of the original positional
    table rather than positions 1..n. Thus the newest observed token always
    receives the same positional vector as the full-trial endpoint.
    """

    def _encode(self, x):
        tok = self.tokenizer
        u = tok.sample_features(x)

        m = u.shape[-1]
        n = (m + tok.pool - 1) // tok.pool
        pad = n * tok.pool - m

        sums = F.pad(u, (0, pad)).reshape(
            u.shape[0], 96, n, tok.pool
        ).sum(-1)

        counts = torch.full(
            (n,),
            tok.pool,
            device=u.device,
            dtype=u.dtype,
        )
        counts[-1] = m - (n - 1) * tok.pool

        avg = sums / counts
        z = tok.project(tok.drop(avg).transpose(1, 2)).transpose(1, 2)

        # Right-align to the full-trial positional table.
        pos = tok.position[:, :, -n:]
        return z + pos

    def forward(self, x):
        z = self._encode(x)
        h = self.forward_reader(z)[:, :, -1]
        return self.classifier(h)


class RandomTruncationForward(_CompactBase):
    """Compact trained with one randomly selected legal duration per minibatch.

    This is NOT a prefix-loss model: every optimizer step still contains one
    ordinary cross-entropy term for one supplied input duration.
    """

    def __init__(self, base, train_lengths):
        super().__init__(base)
        self.train_lengths = tuple(int(x) for x in train_lengths)
        if not self.train_lengths:
            raise ValueError("train_lengths must not be empty")

    def forward(self, x):
        if self.training:
            idx = int(torch.randint(len(self.train_lengths), (1,)).item())
            m = min(self.train_lengths[idx], x.shape[-1])
            x = x[:, :, :m]

        z = self.tokenizer(x)
        h = self.forward_reader(z)[:, :, -1]
        return self.classifier(h)


def make_anchor_control(
    kind,
    *,
    channels,
    classes,
    max_samples,
    pool,
    seed,
    train_lengths,
):
    base = make_model(
        channels=channels,
        classes=classes,
        max_samples=max_samples,
        pool=pool,
        kind="compact",
        seed=seed,
    )

    if kind == "fixed_anchor":
        return FixedAnchorForward(base)

    if kind == "relative_position":
        return RelativePositionForward(base)

    if kind == "random_truncation":
        return RandomTruncationForward(base, train_lengths)

    raise ValueError(kind)
