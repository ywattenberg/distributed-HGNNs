#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=thnn
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/home/%u/distributed-THNN/log/%j.out
#SBATCH --error=/cluster/home/%u/distributed-THNN/log/%j.err
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST:    ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS:    ${SLURM_NTASKS}"

# echo "Data copied at:     $(date)"

# Binary or script to execute
# load modules
module load python/3.11.6

echo "Dependencies installed"

cd $HOME/distributed-THNN/preprocessing

echo "Build finished at:     $(date)"

echo "Starting training at:     $(date)"

# bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Starting training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false"

/cluster/scratch/${USER}/venv/bin/python3 $HOME/distributed-THNN/preprocessing/random_matrices.py

echo "Finished training at:     $(date)"

# discord notification on finish
# bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Finished training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false"

# End the script with exit code 0
exit 0