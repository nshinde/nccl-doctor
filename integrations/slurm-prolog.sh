#!/bin/bash
# Slurm prolog (runs on every node of the job, as root or SlurmdUser).
# Snapshot node + fabric state into the per-job artifact dir before launch.
JOB_DIR="/shared/nccl-doctor/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"
nccl-doctor snapshot --job-dir "$JOB_DIR" --phase start || true
