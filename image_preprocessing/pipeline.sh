#!/bin/bash
# The preprocessing runs on CPUs only; no GPU is needed.
#SBATCH --cpus-per-task=16   # number of CPU cores
#SBATCH --mem=16G           # memory per node (peak was about 11 GB on Killarney)
#SBATCH --time=4:00:00
#SBATCH --output=slogs/%x_%j.out
                            # %x=job-name, %j=job ID
                            # Note: Slurm does not create "slogs"; run "mkdir -p slogs" before sbatch
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=massid45_cleaning

# Stop at the first failing step instead of running later steps on missing data
set -eo pipefail

# Python virtual environment with requirements.txt installed (see README.md);
# override with: sbatch --export=ALL,VENV_DIR=/path/to/venv pipeline.sh
VENV_DIR=${VENV_DIR:-$HOME/virtualenvs/massid45}

# On Digital Research Alliance of Canada clusters (e.g., Vector's Killarney),
# load the modules the environment was built with; elsewhere, skip this
if command -v module >/dev/null 2>&1; then
    module load StdEnv/2023 python/3.10.13 opencv/4.9.0
fi
source "$VENV_DIR/bin/activate"

# Debugging outputs
pwd
python --version
pip freeze

# Location of MassID45 JSON (located in same directory by default)
COCO_JSON_PATH="assets/annots_20250307_coco.json" 

# Directory to construct for storing downloaded/postprocessed data 
DATA_DIR="data/"

# Directory containing annotation tiles (annotation_tiles.zip from Zenodo)
ANNOTATION_TILES_DIR=annotation_tiles

# Directory containing edited bulk images (bulk_images_edited.zip from Zenodo)
EDITED_BULK_IMAGES_DIR=edited

# Name to assign to MassID45 dataset
DATASET_NAME="lifeplan_b_v9"

# Input image size for models
BASE_SLICE_SIZE=1024

# Zoom factor (tile size) to split the bulk images into; default is set to 512 pixels
# Note: zoom factor of 1 = 1024 x 1024, zoom factor of 2 = 512 x 512, zoom factor of 3 = 341 x 341...
ZOOM_FACTOR=2

ACTUAL_SLICE_SIZE=$((BASE_SLICE_SIZE / ZOOM_FACTOR))

# Downloads bulk data from Zenodo
python download_data.py \
    --output_dir $DATA_DIR

# Assembles annotations from annotator tiles
python assemble_annotations.py --input_coco_json $COCO_JSON_PATH \
    --content_root_dir $DATA_DIR \
    --output_dir_processed_data ${DATA_DIR}/preprocessing_output/ \
    --final_output_coco_dir ${DATA_DIR}/${DATASET_NAME} \
    --annotation_tiles_dir_name ${ANNOTATION_TILES_DIR} \
    --bulk_images_dir_name ${EDITED_BULK_IMAGES_DIR} 

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
    --input_base_dir ${DATA_DIR}/${DATASET_NAME}_cropped \
    --sahi_output_base_dir ${DATA_DIR}/sahi_datasets \
    --zoom_factors ${ZOOM_FACTOR} \
    --base_slice_size ${BASE_SLICE_SIZE} \
    --overlap_ratio 0.6 \
    --min_area_ratio 0.1

# Note: only run the scripts below if you specified one zoom factor in the $ZOOM_FACTOR variable;
# otherwise, run these separately for each zoom factor

# Postprocess invalid and/or disjoint polygons after tiling
python postprocess_dataset.py --dataset_path ${DATA_DIR}/sahi_datasets/sahi_${ACTUAL_SLICE_SIZE}_keep_neg

# Check that all resulting annotations are valid
python analyze_invalid_polygons.py --dataset_path ${DATA_DIR}/sahi_datasets/sahi_${ACTUAL_SLICE_SIZE}_keep_neg