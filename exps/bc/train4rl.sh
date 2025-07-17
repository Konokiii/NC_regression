#!/bin/bash
#SBATCH --verbose
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --partition=nvidia
#SBATCH --mem=32GB
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zd662@nyu.edu
#SBATCH --output=./logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
#SBATCH --error=./logs/%A_%a.err # MAKE SURE WHEN YOU RUN THIS, ../logs IS A VALID PATH
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

#echo "SLURM_JOBID: $SLURM_JOBID"

echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

nvidia-smi

source activate common_dl

cd $SCRATCH/NC_regression
export PYTHONPATH=$PYTHONPATH:/scratch/zd662/NC_regression
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zd662/.mujoco/mujoco210/bin:/usr/lib/nvidia

# Launch parallel jobs in background
for i in {0..9}
do
  index=$(( 10 * SLURM_ARRAY_TASK_ID + i ))
  individual_log="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${index}"

  echo "Launching job $index"

  python exps/bc/train4rl.py --setting $index > exps/logs/${individual_log}.out 2> exps/logs/${individual_log}.err &

done

# Wait for all background jobs to complete
wait

echo "All jobs completed."


#for i in {0..15}
#do
#  echo "Launching job $i"
#  python main/exp/iclr2025/collect.py --setting $i > main/exp/logs/${SLURM_JOBID}_${i}.out 2> main/exp/logs/${SLURM_JOBID}_${i}.err &
#done
#
## Wait for all background jobs to complete
#wait
#
#echo "All jobs completed."