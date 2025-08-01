#!/bin/bash
#SBATCH --verbose
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=nvidia
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=zd662@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-12 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=./logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
#SBATCH --error=./logs/%A_%a.err # MAKE SURE WHEN YOU RUN THIS, ../train_logs IS A VALID PATH

# #####################################################
#SBATCH --gres=gpu:1 # uncomment this line to request a gpu
#SBATCH --cpus-per-task=8

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

nvidia-smi

for k in {0..4}
do
  setting1=$((10 * SLURM_ARRAY_TASK_ID + 2 * k))
  setting2=$((10 * SLURM_ARRAY_TASK_ID + 2 * k + 1))

  singularity exec --nv \
  -B /$SCRATCH/NC_regression:/NC_regression \
  -B /$SCRATCH/corl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ \
  /$SCRATCH/corl-sandbox bash -c "
  cd /NC_regression
  export PYTHONPATH=\$PYTHONPATH:/NC_regression
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/zd662/.mujoco/mujoco210/bin:/usr/lib/nvidia
  python nrc_manifold/mse_evaluation.py --setting $setting1 \
  > nrc_manifold/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${setting1}.out \
  2> nrc_manifold/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${setting1}.err
  " &

  singularity exec --nv \
  -B /$SCRATCH/NC_regression:/NC_regression \
  -B /$SCRATCH/corl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ \
  /$SCRATCH/corl-sandbox bash -c "
  cd /NC_regression
  export PYTHONPATH=\$PYTHONPATH:/NC_regression
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/zd662/.mujoco/mujoco210/bin:/usr/lib/nvidia
  python nrc_manifold/mse_evaluation.py --setting $setting2 \
  > nrc_manifold/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${setting2}.out \
  2> nrc_manifold/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${setting2}.err
  " &

  wait

done

#export PYTHONPATH=$PYTHONPATH:/NC_regression
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zd662/.mujoco/mujoco210/bin:/usr/lib/nvidia