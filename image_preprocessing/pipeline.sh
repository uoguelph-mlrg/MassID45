#!/bin/bash
#SBATCH --gres=gpu:l40s:1   # request GPU(s)
#SBATCH --cpus-per-task=16   # number of CPU cores
#SBATCH --mem=12G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --time=4:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=massid45_cleaning

ENV_NAME=mid45_testing
module load StdEnv/2020 gcc/9.3.0
module load cuda/11.8.0
module load python/3.10.2
module load opencv/4.8.0
source /home/jquinto/projects/aip-gwtaylor/jquinto/virtualenvs/$ENV_NAME/bin/activate
SEED="$SLURM_ARRAY_TASK_ID"

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
    --input_base_dir data/${DATASET_NAME}_cropped \
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