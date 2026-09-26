#!/bin/bash
set -euo pipefail

RUN="$1"
ORIG="$2"
PY="/users/nikkieh/bibm_env/bin/python"

cd "$RUN/code"
export PYTHONPATH="$PWD"

"$PY" -u -m fresh.anchor_report \
    --orig-study "$ORIG" \
    --diag-study "$RUN" \
    > "$RUN/anchor_summary.txt"

echo "Summary:"
cat "$RUN/anchor_summary.txt"
