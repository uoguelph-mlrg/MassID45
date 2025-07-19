#!/bin/bash
#SBATCH -p cpu        # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH -c 32              # number of CPU cores
#SBATCH --mem=20G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=cpu_qos
#SBATCH --time=24:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=MassID45_image_cleaning_pipeline

source ~/.bashrc
source activate massid45

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# Location of MassID45 JSON (in same directory by default)
COCO_JSON_PATH="annots_20250307_coco.json" 

# Directory to construct for storing downloaded/postprocessed data 
DATA_DIR="data/"

# Name to assign to MassID45 dataset
DATASET_NAME="lifeplan_b_v9_tsting"

# Input image size for models
BASE_SLICE_SIZE=1024

# Zoom factor (tile size) to split the bulk images into; default is set to 512 pixels
# Note: zoom factor of 1 = 1024 x 1024, zoom factor of 2 = 512 x 512, zoom factor of 3 = 341 x 341...
ZOOM_FACTOR=2

ACTUAL_SLICE_SIZE=$((BASE_SLICE_SIZE / ZOOM_FACTOR))

# Downloads bulk data from GDrive
python download_data.py \
    --output_dir $DATA_DIR

# Assembles annotations from annotator tiles
python assemble_annotations.py \
    --input_coco_json $COCO_JSON_PATH \
    --content_root_dir $DATA_DIR \
    --output_dir_processed_data ${DATA_DIR}/preprocessing_output/ \
    --final_output_coco_dir ${DATA_DIR}/${DATASET_NAME} \
    # --save_intermediate_images

# Crops bulk images down to area containing insects, with some buffer
python crop_bulk_imgs.py \
  --input_base_dir ${DATA_DIR}/${DATASET_NAME} \
  --output_dir_cropped_dataset ${DATA_DIR}/${DATASET_NAME}_cropped \
  --buffer_pixels 200 \
  --visualize_calculated_bboxes \
  --plot_masks_on_bbox_viz \
  --visualize_cropped_annotations \
  --print_calculated_bboxes

# Divides bulk images into tiles using SAHI
python tile_imgs.py \
    --input_base_dir data/${DATASET_NAME}_cropped \
    --sahi_output_base_dir ${DATA_DIR}/sahi_datasets \
    --zoom_factors ${ZOOM_FACTOR} \
    --base_slice_size ${BASE_SLICE_SIZE} \
    --overlap_ratio 0.6 \
    --min_area_ratio 0.1

# # Note: only run the scripts below if you specified one zoom factor in the $ZOOM_FACTOR variable;
# # otherwise, run these separately for each zoom factor

# Postprocess invalid and/or disjoint polygons after tiling
python postprocess_dataset.py --dataset_path ${DATA_DIR}/sahi_datasets/sahi_${ACTUAL_SLICE_SIZE}_keep_neg

# Check that all resulting annotations are valid
python analyze_invalid_polygons.py --dataset_path ${DATA_DIR}/sahi_datasets/sahi_${ACTUAL_SLICE_SIZE}_keep_neg