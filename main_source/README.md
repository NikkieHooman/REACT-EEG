# READER fresh matched study

**This package trains NEW models. It does not reproduce the historical manuscript
numbers and does not load or overwrite any legacy checkpoint.** The fresh model
implements the manuscript's plain causal front end, partial-window mean pooling,
zero-residual initialization, and endpoint-only objective. Choices that were not
fully specified in the manuscript are fixed explicitly in `docs/PROTOCOL.md`.

No EEG measurements are bundled. Local validation uses synthetic arrays only.
The real experiment starts only when the user runs `launch.sh` in their existing
SMU HPC session. The chat does not have authenticated remote execution.

## One launch, not another discovery cycle

Existing locations are used directly:

- Python: `/users/nikkieh/bibm_env/bin/python`
- 2A: `/lustre/smuexa01/client/users/nikkieh/bibm_data/bci/2a/2a_pre`
- 2B: `/lustre/smuexa01/client/users/nikkieh/bibm_data/bci/2b/2b_pre`
- SSVEP: `/lustre/smuexa01/client/users/nikkieh/bibm_data/sd_ssvep/processed_kaggle_exact`
- Slurm: `batch`, account `eclarson_ehr_fair_0003`

After extracting this package on the HPC:

```bash
bash reader_fresh_study/launch.sh
```

This freezes the code and protocol into a fresh directory under
`/lustre/smuexa01/client/users/nikkieh/reader_fresh_runs/`. It first resolves
BiTE commit `924eb32241ba1a7c80dbc4ba097f8c979da17578` from the existing local
BiteEEG Git objects, or fetches that exact commit into the new study. It never
updates the old checkout or silently falls back to current upstream code.
No `pip install` or environment upgrade is performed.

Then it submits a training-role-only GPU preflight, an after-success array of 84
subject/seed jobs, and an after-any-result CPU collection job. Each array job
trains Compact, READER, FF, Compact--Mean, and BiTE sequentially on one GPU:
**420 total training runs of 600 epochs**. Default concurrency is four GPUs.
Optional: `READER_CONCURRENT_GPUS=2 bash launch.sh` caps concurrent jobs at two.

The preflight actually imports the pinned BiTE code and performs training steps
on training data before the large array can start. Local tests could not fetch
or execute that actual upstream revision; this is explicitly not a claim of
local BiTE validation. A dependency or API failure stops the gate and is returned
as an error, not hidden with a replacement baseline.

## Monitor

```bash
RUN="$(cat /users/nikkieh/reader_followup/session_B1VqC0/latest_fresh_study.txt)"
bash "$RUN/code/status.sh" "$RUN"
```

Do not launch a new study to resume an interrupted one. Once every old training
and collection job has ended, use:

```bash
bash "$RUN/code/resume_incomplete.sh" "$RUN"
```

Resume requires unchanged code/data/recipe and matching saved software/GPU-type
metadata. It restores optimizer, scheduler and random states, skipping completed
models. It does not shorten the 600-epoch horizon. A failed correctness test is
not a reason to edit result metadata or disable the check.

## Outputs

Each model retains `final.pt`, resumable `last.pt`, training history, per-trial
predictions with IDs and actual sample lengths, and metadata hashes. Each
subject/seed group retains its training-only fitted scaler and role hashes.

On completion the collector checks all 420 results, recomputes accuracies from
saved logits, checks checksums and pairing, then creates:

- `analysis/`: seed-averaged subject means, paired bootstrap contrasts, endpoint
  and normalized accuracy-duration summaries, all core-model curves;
- `paper/`: the revised manuscript with measured tables and result text; no old
  numeric tables are imported;
- `return_outputs.zip`: status, source snapshot, small run records, analyses and
  manuscript. Large checkpoints, raw EEG and prediction-array files remain on
  the HPC and are not included in the return bundle.

If LaTeX/BibTeX are installed, `paper/main.pdf` is compiled. Otherwise all `.tex`,
`.bib`, table and figure files are returned for local/Overleaf compilation.
Compilation does not establish publication readiness.

`cohort_status.json` distinguishes INCOMPLETE, REPORT_ERROR,
MEASURED_CAUSALITY_REVIEW_REQUIRED, and MEASURED_COHORT_COMPLETE. A complete
cohort still returns paper status `RESULTS_FILLED_REQUIRES_AUTHOR_REVIEW`.
No author identity, provenance approval or conference compliance is invented.

## Submission boundary

Author details, raw-to-prepared provenance, declared benchmark exposure,
interpretation, reference metadata, workshop page limit/anonymization, and the
final visual PDF need review. A new fit is **not** a previously unseen test set:
the original benchmark results were already inspected. This disclosure is in
the generated manuscript.

The legacy HGD, LOSO and specialist tables are excluded from this fresh study.
They cannot be reused automatically because their provenance is unresolved.
See `docs/MANUSCRIPT_CHANGES.md` for exactly what is retained and replaced.

## Validate locally before freezing

```bash
PYTHONPATH=. python -m pytest tests -q
```

Do not run package-writing test fixtures inside an already frozen HPC code
snapshot. They write only clearly labelled software-validation records; no
synthetic result file is eligible for the real-paper collector.
