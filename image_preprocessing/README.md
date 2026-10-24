
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
        └── annotation_tiles              # Raw data (annotation_tiles.zip on Zenodo)
        └── edited                        # Full bulk images (bulk_images_edited.zip on Zenodo)
        └── preprocessing_output          # output of preprocessing scripts
        └── lifeplan_b_v9                 # Uncropped, assembled bulk images
        └── lifeplan_b_v9_cropped         # Cropped, assembled bulk images
        └── sahi_datasets                 # Tiled bulk images -- for ML models
            └── sahi_512_keep_neg             

   ├── annots_20250307_coco.json          # Dataset JSON for MassID45 (provided in release)
   └── pipeline.sh                        # Main bash script for running preprocessing pipeline
```
### Installation
The pipeline needs only Python and the packages in `requirements.txt`; it runs on CPUs and does not need a GPU. The steps below were tested on Vector's Killarney cluster, where we developed MassID45, and should work on other Digital Research Alliance of Canada clusters. On any other machine, use your own paths and environment manager.

Clone the repository (the submodules are not needed for preprocessing):
```bash
git clone https://github.com/uoguelph-mlrg/MassID45.git
cd MassID45/image_preprocessing
```

### Install dependencies
On a machine with internet access and no module system:
```bash
python3 -m venv ~/virtualenvs/massid45
source ~/virtualenvs/massid45/bin/activate
pip install -r requirements.txt
```

On Alliance clusters, build the environment from the Alliance wheelhouse. `sahi==0.11.18` is not in the wheelhouse, and its `opencv-python` requirement conflicts with the OpenCV module, so install it from PyPI without dependencies (the login node has internet access):
```bash
module load StdEnv/2023 python/3.10.13 opencv/4.9.0
virtualenv --no-download ~/virtualenvs/massid45
source ~/virtualenvs/massid45/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch "numpy<2" matplotlib Pillow shapely tqdm requests pyyaml fire terminaltables click
pip download --no-deps --dest wheels sahi==0.11.18 pybboxes==0.1.6
pip install --no-index --no-deps --find-links wheels sahi==0.11.18 pybboxes==0.1.6
```

### How to Run
`pipeline.sh` activates the environment in `VENV_DIR` (default `~/virtualenvs/massid45`) and, where the Alliance module system is present, loads the modules above. Slurm writes its log to `slogs/`, which must exist before you submit.

With Slurm:
```bash
mkdir -p slogs
sbatch pipeline.sh
# or, with an environment elsewhere:
sbatch --export=ALL,VENV_DIR=/path/to/venv pipeline.sh
```
On Killarney, submit from a directory under `/scratch` (for example, clone into `$SCRATCH`): Slurm refuses jobs submitted from `/home`. The run takes about an hour and peaks at about 16 GB of memory.

Without Slurm, run it directly: `bash pipeline.sh`.
