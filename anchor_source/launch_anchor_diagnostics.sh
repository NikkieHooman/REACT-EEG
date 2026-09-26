#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

ORIG="/lustre/smuexa01/client/users/nikkieh/reader_fresh_runs/study_20260918T005712Z_qDm6ij"
PARENT="/lustre/smuexa01/client/users/nikkieh/reader_anchor_diagnostics"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$PARENT/study_${STAMP}_$$"

mkdir -p \
    "$RUN/code" \
    "$RUN/logs" \
    "$RUN/runs"

cp -a "$HERE"/. "$RUN/code/"

cat > "$RUN/PROVENANCE.txt" <<EOF
Original paper study:
$ORIG

Diagnostic source:
$HERE

Purpose:
Fixed-anchor, right-aligned-position, random-truncation, reverse-subset,
and reverse-gradient diagnostics.

Original paper outputs are read-only and are not modified.
EOF

COMMON=(
  --parsable
  --partition=batch
  --account=eclarson_ehr_fair_0003
  --chdir="$RUN/code"
)

PREFLIGHT="$(
  sbatch "${COMMON[@]}" \
    --job-name=react_anchor_preflight \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:20:00 \
    --output="$RUN/logs/preflight.out" \
    --error="$RUN/logs/preflight.out" \
    scripts/anchor_preflight.sbatch \
    "$RUN"
)"

ARRAY="$(
  sbatch "${COMMON[@]}" \
    --job-name=react_anchor \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=12:00:00 \
    --array="0-83%4" \
    --dependency="afterok:$PREFLIGHT" \
    --kill-on-invalid-dep=yes \
    --output="$RUN/logs/task_%A_%a.out" \
    --error="$RUN/logs/task_%A_%a.out" \
    scripts/anchor_train.sbatch \
    "$RUN" \
    "$ORIG"
)"

COLLECT="$(
  sbatch "${COMMON[@]}" \
    --job-name=react_anchor_report \
    --cpus-per-task=2 \
    --mem=12G \
    --time=00:30:00 \
    --dependency="afterok:$ARRAY" \
    --output="$RUN/logs/collect.out" \
    --error="$RUN/logs/collect.out" \
    scripts/anchor_collect.sbatch \
    "$RUN" \
    "$ORIG"
)"

cat > "$RUN/jobs.txt" <<EOF
RUN=$RUN
PREFLIGHT=$PREFLIGHT
ARRAY=$ARRAY
COLLECT=$COLLECT
EOF

echo "RUN=$RUN"
echo "PREFLIGHT=$PREFLIGHT"
echo "ARRAY=$ARRAY"
echo "COLLECT=$COLLECT"
echo
echo "Monitor:"
echo "  squeue -j $PREFLIGHT,$ARRAY,$COLLECT"
echo
echo "Summary after completion:"
echo "  cat $RUN/anchor_summary.txt"
echo "  cat $RUN/anchor_summary.json"
