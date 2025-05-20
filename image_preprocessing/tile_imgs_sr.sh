#!/bin/bash
#SBATCH -p cpu        # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH -c 32              # number of CPU cores
#SBATCH --mem=32G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=cpu_qos
#SBATCH --time=24:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=SAHI_dataset_tile_splitting_2x_zoom_SR_tiles_png

source ~/.bashrc
source activate md3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

python tile_imgs_sr.py

