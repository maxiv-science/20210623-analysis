#!/bin/bash
#SBATCH --job-name=sharonb
#SBATCH --output={job_file}.out
#SBATCH --error={job_file}.err
#SBATCH --partition=all
#SBATCH --exclusive
#SBATCH --time 12:00:00

echo "SLURM_JOB_ID           $SLURM_JOB_ID"

. /home/sharonb/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate test

python slurm_ttc_analysis.py {1} $SLURM_JOB_ID    



