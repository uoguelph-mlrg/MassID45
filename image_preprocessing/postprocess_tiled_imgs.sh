#!/bin/bash
#SBATCH -p cpu        # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH -c 16              # number of CPU cores
#SBATCH --mem=32G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=cpu_qos
#SBATCH --time=24:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=SAHI_dataset_tile_postprocessing_dataset_zoom_factors_8x

source ~/.bashrc
source activate md3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_1536_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_768_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_512_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_384_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_307_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_256_ignore_neg

# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_1024_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_512_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_341_ignore_neg

# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_256_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_170_ignore_neg
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets/sahi_128_ignore_neg

# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_bicubic/sahi_1024_ignore_neg_SR
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_real_esrgan/sahi_1024_ignore_neg_SR
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_swinir/sahi_1024_ignore_neg_SR
# python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_hat-l/sahi_1024_ignore_neg_SR
python postprocess_dataset.py --dataset_path /h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_swinir_bioscan/sahi_1024_ignore_neg_SR









