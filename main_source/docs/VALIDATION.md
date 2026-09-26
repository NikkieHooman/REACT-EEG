# Validation performed before delivery

**These are local software checks, not new EEG benchmark results.**

42 tests passed in three completed test invocations:
- 35 model/data/training/causality and mocked Slurm-launch tests;
- 5 complete-report, rejection and synthetic-layout tests;
- 2 synthetic end-to-end integration and timing-function tests.

The integration test trained the four new core models on synthetic arrays at
all three configured input shapes: 12 training runs, two epochs each. It checked
that held-role loading occurred only after all core models in that group were
trained, generated exact-prefix predictions, ran trained and nondegenerate causal
checks, and verified that a completed-group rerun did not alter final weights.
The collector explicitly refused to produce a research manuscript from synthetic
results. Interrupted training also reproduced an uninterrupted synthetic run's
weights and optimizer trajectory under the same local environment.

Other checks covered partial-pool arithmetic, identity initialization, matched
shared parameters, the 3,136 added-parameter relationship, role overlap, no label
coercion, source/plan changes, a deliberately leaking computation, missing results,
checkpoint/prediction tampering, and false epoch metadata.

The report-generator test compiled a **visibly labelled synthetic layout test**
with IEEEtran/LaTeX. It produced six pages with no overfull-box diagnostics. The
layout was inspected through rendered pages. Its fake numbers and generated PDF
are NOT distributed and are NOT evidence for the real manuscript. Real results,
author details and final float positions still require a fresh visual review.

Local versions are recorded in `validation/environment.json`. Local PyTorch is
2.10.0+cpu; the known HPC environment is 2.5.1+cu124. There was no local GPU or
access to the HPC EEG arrays. The pinned actual upstream BiTE revision could not
be fetched/executed in this environment; its adapter follows the supplied code,
and actual import/gradient validation is enforced by the HPC preflight before
any 600-epoch array task is eligible to run. No success on that preflight is
claimed before its output is returned.

Mocked Slurm tests check the dependency graph and resource arguments; they are
not remote submissions. No HPC jobs have been submitted from this chat session.
