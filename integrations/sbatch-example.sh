#!/bin/bash
#SBATCH -J llama-70b-pretrain
#SBATCH -N 4 --gpus-per-node=8
# User-facing path when prolog/epilog integration isn't deployed:
JOB_DIR="/shared/nccl-doctor/${SLURM_JOB_ID}"
srun --ntasks-per-node=1 nccl-doctor snapshot --job-dir "$JOB_DIR" --phase start
nccl-doctor run --job-dir "$JOB_DIR" --job-name "$SLURM_JOB_NAME" --fabric ib -- \
    torchrun --nnodes "$SLURM_NNODES" --nproc-per-node 8 train.py "$@"
RC=$?
srun --ntasks-per-node=1 nccl-doctor snapshot --job-dir "$JOB_DIR" --phase end
[ $RC -ne 0 ] && nccl-doctor analyze "$JOB_DIR" --store /shared/nccl-doctor/store.db
exit $RC
