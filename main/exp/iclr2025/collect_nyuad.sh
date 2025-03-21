#!/bin/bash
#SBATCH --verbose
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=nvidia
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zd662@nyu.edu
#SBATCH --output=../logs/%j_%x.out
#SBATCH --error=../logs/%j_%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18

echo "SLURM_JOBID: $SLURM_JOBID"

source activate otr

cd $SCRATCH/NC_regression
export PYTHONPATH=$PYTHONPATH:/scratch/zd662/NC_regression
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zd662/.mujoco/mujoco210/bin:/usr/lib/nvidia

# Launch parallel jobs in background
for i in {0..8}
do
  echo "Launching job $i"
  python main/exp/iclr2025/collect.py --setting $i > ../logs/${SLURM_JOBID}_${i}.out 2> ../logs/${SLURM_JOBID}_${i}.err &
done

# Wait for all background jobs to complete
wait

echo "All jobs completed."