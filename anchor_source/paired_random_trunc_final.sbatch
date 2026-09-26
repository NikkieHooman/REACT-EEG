#!/usr/bin/env bash
#SBATCH -J react_rt_pair
#SBATCH -A eclarson_ehr_fair_0003
#SBATCH -p short
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-83%12
#SBATCH -o /lustre/smuexa01/client/users/nikkieh/reader_random_trunc_paired_final/logs/%A_%a.out
#SBATCH -e /lustre/smuexa01/client/users/nikkieh/reader_random_trunc_paired_final/logs/%A_%a.err

set -euo pipefail

PY="/users/nikkieh/bibm_env/bin/python"

ORIG="/lustre/smuexa01/client/users/nikkieh/reader_fresh_runs/study_20260918T005712Z_qDm6ij"

PROJ="/home/nikkieh/projects/REACT_anchor_diag_20260920_214124"

OUT="/lustre/smuexa01/client/users/nikkieh/reader_random_trunc_paired_final"

mkdir -p "$OUT/logs"

srun "$PY" \
  "$PROJ/train_paired_random_trunc_final.py" \
  --study "$ORIG" \
  --out "$OUT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --device cuda
