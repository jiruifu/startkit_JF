#!/bin/bash

#SBATCH --job-name=test_ddp   # Job name
#SBATCH --output=test_ddp_%j.out  # Name of output file (%j expands to jobId)
#SBATCH --error=test_ddp_%j.err   # Name of error file (%j expands to jobId)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=2        # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --time=01:00:00            # Time limit hrs:min:sec
#SBATCH --gres=gpu:2               # Number of GPUs per node
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ji045311@ucf.edu  # Where to send mail

echo "==================================================================="
echo "Job started at: $(date)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "==================================================================="

# Load necessary modules (modify according to your system)
module purge
module load anaconda/anaconda-2023.09
conda activate eeg2025

# Navigate to the script directory
cd /lustre/fs1/home/ji045311/EEG_Challenge/

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "==================================================================="

# Run the distributed training script with torchrun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    test_ddp.py \
    --model resnet18 \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --save_dir /lustre/fs1/home/ji045311/EEG_Challenge_Output/ddp_test

echo "==================================================================="
echo "Job finished at: $(date)"
echo "==================================================================="

