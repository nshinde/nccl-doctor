#!/bin/bash
# Slurm epilog: end snapshot on every node; analysis once (lowest-numbered node).
JOB_DIR="/shared/nccl-doctor/${SLURM_JOB_ID}"
[ -d "$JOB_DIR" ] || exit 0
nccl-doctor snapshot --job-dir "$JOB_DIR" --phase end || true
FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
if [ "$(hostname -s)" = "$FIRST_NODE" ]; then
  # convert torch flight-recorder dumps if the job produced them
  python3 /opt/nccl-doctor/tools/fr_export.py \
      "$JOB_DIR"/flight_recorder/torch_nccl_trace_* \
      --out "$JOB_DIR/flight_recorder" 2>/dev/null || true
  nccl-doctor analyze "$JOB_DIR" \
      --store /shared/nccl-doctor/store.db \
      --out "$JOB_DIR/report.json" > "$JOB_DIR/report.txt" || true
fi
