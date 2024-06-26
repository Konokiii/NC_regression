#!/bin/bash
#SBATCH --verbose
#SBATCH --time=24:00:00
# #SBATCH --nodes=1
#SBATCH --exclude=gm[002-020],gm[022-023],gm024
#SBATCH --mem=48GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=zd662@nyu.edu # NOTE: put your netid here if you want emails

# #SBATCH --array=0-9 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=../logs/%A.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
#SBATCH --error=../logs/%A.err # MAKE SURE WHEN YOU RUN THIS, ../train_logs IS A VALID PATH

# #####################################################
#SBATCH --gres=gpu:rtx8000:1 # uncomment this line to request a gpu
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4


sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

singularity exec --nv -B /scratch/$USER/NC_regression:/NC_regression -B /scratch/$USER/cql-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ /scratch/$USER/cql-sandbox bash -c "
cd /NC_regression
export PYTHONPATH=$PYTHONPATH:/NC_regression
nvidia-smi
python main/exp/case1/gamma.py --setting 0 &
python main/exp/case1/gamma.py --setting 1 &
python main/exp/case1/gamma.py --setting 2 &
wait"