#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=fwd-timing
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/home/%u/distributed-THNN/log/%j.out
#SBATCH --error=/cluster/home/%u/distributed-THNN/log/%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST:    ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS:    ${SLURM_NTASKS}"

rsync -ah --stats /cluster/home/$USER/distributed-THNN/data $TMPDIR
rsync -ah --stats /cluster/scratch/$USER/data/random $TMPDIR/data

# echo "Data copied at:     $(date)"

# Binary or script to execute
# load modules
module load gcc/11.4.0 openmpi openblas cmake/3.26.3 eth_proxy curl

echo "Dependencies installed"

cd $HOME/distributed-THNN/build
cmake ..
make -j ${SLURM_NTASKS}

echo "Build finished at:     $(date)"

echo "Starting timing run at:     $(date)"

# bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Starting training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false"

$HOME/distributed-THNN/build/forward_timing -d $TMPDIR -i ${SLURM_JOB_ID} -p ${SLURM_NTASKS}

echo "Finished timing run at:     $(date)"

# discord notification on finish
# bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Finished training for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false"

# End the script with exit code 0
exit 0