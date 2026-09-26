"""Generic controls using supplied original modules; canonical token shape is B,D,N.

The wrapper does not assume the original repository's internal attribute names.
A components factory constructs tokenizer, forward_reader, classifier in the
historical order and returns a separate zero-argument reader_factory.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .common import resolve


class ReadoutControl(nn.Module):
    def __init__(self, tokenizer: nn.Module, forward_reader: nn.Module,
                 classifier: nn.Module, reader_factory: Callable[[], nn.Module],
                 kind: str, width: int = 64, extra_seed: int = 918273):
        super().__init__()
        if kind not in {"reader", "compact", "ff", "compact_mean"}:
            raise ValueError(kind)
        self.tokenizer, self.forward_reader, self.classifier = tokenizer, forward_reader, classifier
        self.kind, self.width = kind, width
        if kind in {"reader", "ff"}:
            # All common modules have already been initialized. CPU constructor
            # randomness for the additional branch does not consume common RNG.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(extra_seed)
                self.second_reader = reader_factory()
            self.gate_logits = nn.Parameter(torch.zeros(width))
        else:
            self.second_reader = None
            self.register_parameter("gate_logits", None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tokenizer(x)
        if z.ndim != 3 or z.shape[1] != self.width:
            raise ValueError("Tokenizer must return positioned [B,width,N] tokens")
        return z

    def states(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        f_seq = self.forward_reader(z)
        if f_seq.shape != z.shape:
            raise ValueError("TCN must preserve [B,D,N]")
        f = f_seq[:, :, -1]
        b = None
        if self.kind == "compact_mean":
            h = f_seq.mean(dim=-1)  # Equal token weighting, including partial token.
        elif self.kind == "compact":
            h = f
        else:
            zin = z.flip(-1) if self.kind == "reader" else z
            b = self.second_reader(zin)[:, :, -1]
            g = self.gate_logits.sigmoid()
            h = (1-g) * f + g * b
        return f, b, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.states(self.encode(x))[2])

    def branch_diagnostics(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        f, b, h = self.states(z)
        if b is None:
            return {"kind": self.kind, "tokens": z.shape[-1]}
        g = self.gate_logits.sigmoid()
        # A controlled in-prefix intervention at the SECOND reader input.
        # Exclude original token 1 so temporal use is distinguished from z1-only.
        zp = z.clone()
        if z.shape[-1] > 1:
            delta = 0.1 * z.detach().std().clamp_min(1e-4)
            zp[:, :, 1:] = zp[:, :, 1:] + delta
        _, bp, _ = self.states(zp)
        base = self.classifier(h)
        f_logits, b_logits = self.classifier(f), self.classifier(b)
        return {"kind": self.kind, "tokens": int(z.shape[-1]), "gate": g,
                "forward_norm_mean": f.norm(dim=1).mean(), "reverse_norm_mean": b.norm(dim=1).mean(),
                "weighted_forward_norm_mean": ((1-g)*f).norm(dim=1).mean(),
                "weighted_reverse_norm_mean": (g*b).norm(dim=1).mean(),
                "branch_cosine_mean": torch.nn.functional.cosine_similarity(f,b,dim=1).mean(),
                "in_prefix_context_intervention_max": (b-bp).abs().max(),
                "full_logits": base, "forward_only_posthoc_logits": f_logits,
                "reverse_only_posthoc_logits": b_logits,
                "interpretation": "posthoc shared-classifier interventions, NOT separately trained controls"}

    @torch.no_grad()
    def project_constraints(self):
        # Apply only a supplied tokenizer's stated constraint operation.
        if hasattr(self.tokenizer, "project_constraints"):
            self.tokenizer.project_constraints()
        if hasattr(self.classifier, "project_constraints"):
            self.classifier.project_constraints()


def build_control(components_factory: str, kind: str, seed: int,
                  extra_seed: int = 918273, components_kwargs: dict | None = None) -> ReadoutControl:
    """The components factory must reuse the user's actual modules, not placeholders."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        c = resolve(components_factory)(**(components_kwargs or {}))
    needed = {"tokenizer", "forward_reader", "classifier", "reader_factory"}
    if needed - set(c):
        raise ValueError(f"components_factory must return {sorted(needed)}")
    return ReadoutControl(c["tokenizer"], c["forward_reader"], c["classifier"],
                          c["reader_factory"], kind, c.get("width",64), extra_seed)


def components_from_compact(compact_factory:str,compact_kwargs:dict,tokenizer_path:str,
                            forward_reader_path:str,classifier_path:str,
                            extra_reader_factory:str,extra_reader_kwargs:dict|None=None,
                            width:int=64) -> dict:
    """Extract audited modules from the ACTUAL Compact implementation.

    tokenizer_path must include projection AND original positional embeddings and
    return [B,D,N]. If the source splits those operations across methods, write a
    small project-specific components factory instead of guessing a module path.
    This constructs a fresh Compact, not a trained checkpoint ablation.
    """
    from functools import partial
    compact=resolve(compact_factory)(**compact_kwargs)
    return {"tokenizer":compact.get_submodule(tokenizer_path),
            "forward_reader":compact.get_submodule(forward_reader_path),
            "classifier":compact.get_submodule(classifier_path),
            "reader_factory":partial(resolve(extra_reader_factory),**(extra_reader_kwargs or {})),
            "width":width}
