
# MassID45 Image Preprocessing

This directory contains scripts and utilities for cleaning and processing the MassID45 project data. The main script, `pipeline.sh`, downloads the MassID45 bulk images, processes image patches, and prepares the dataset for further analysis or machine learning tasks like segmentation. The folder structure and paths have been set up to work with a specific dataset organization. The dataset JSON `annots_20250307_coco.json` is included in this directory in COCO format. 

Main tasks performed by this folder:
1. Loads MassID45 Annotations: Reads and processes the annotations in JSON format.
2. Assemble Image Patches: Processes and assembles image tiles from the raw data.
3. Cropping: crops the edges of bulk images where no insects are present.
4. Tiling: Assembled images are divided into tiles of fixed size with overlap.
5. Preprocessing: Merges annotation polygons as required and validates invalid polygons. Correct invalid segmentation masks.
7. Splits the Dataset: Creates train/test/validation splits of the dataset.

## Folder Structure

The expected structure for this directory after running `pipeline.sh` is as follows:

```bash
image_preprocessing
   ├── data                               # Folder created by pipeline.sh
        └── annotated_tiles               # Raw data (annotated_tiles.zip on Zenodo)
        └── edited                        # Full bulk images (edited.zip on Zenodo)     
        └── preprocessing_output          # output of preprocessing scripts
        └── lifeplan_b_v9                 # Uncropped, assembled bulk images
        └── lifeplan_b_v9_cropped         # Cropped, assembled bulk images
        └── sahi_datasets                 # Tiled bulk images -- for ML models
            └── sahi_512_keep_neg             

   ├── annots_20250307_coco.json          # Dataset JSON for MassID45 (provided in release)
   └── pipeline.sh                        # Main bash script for running preprocessing pipeline
```
### Installation
Follow the instructions to below to use the package on the Killarney (or similar Compute Canada) cluster.

Steps:
Clone the repository:
```bash
git clone --recurse-submodules git@github.com:Jquinto64/MassID45.git
cd MassID45
```

### Install dependencies:
Make sure you have pip installed. Then run the following command to install the required Python packages:
```bash
module load StdEnv/2020  gcc/9.3.0
module load python/3.10.2
module load cuda/11.8.0
module load opencv/4.8.0

ENV_NAME=<your_env_name_here>
virtualenv --no-download virtualenvs/$ENV_NAME
source ~/virtualenvs/$ENV_NAME/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch numpy matplotlib pandas
pip install -r requirements.txt
```
This will install all the necessary libraries needed for the script to run.

### How to Run
Once the dependencies are installed, you can run the pipeline.sh script to process the data.

Usage:
```bash
sbatch pipeline.sh
```