# MassID45
This repo contains the preprocessing and training/inference code for MassID45: [A multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level](https://arxiv.org/abs/2507.06972).

# Installation
Requirements for downloading and preprocessing the data can be found in the `image_preprocessing` folder. We recommend creating a dedicated Python environment for `image_preprocessing`.

The `CutLER`, `Mask-RCNN`, `Mask2Former`, `MaskDINO`, and `Grounded-SAM-2` submodules each contain their own installation instructions (see the `INSTALL.md` files in each submodule). We recommend creating separate Python or Conda environments for each submodule.

# Usage

## Annotation utilities
Code to run the watershed algorithm and crop the bulk images into annotator patches is contained in the `arth-imrec` submodule.

## Downloading bulk imagery data
The bulk images can be downloaded from Google Drive by running `image_preprocessing/download_data.py`. Scripts for downloading the ENA sequence data can be found in the `image_preprocessing/ENA_sequence_scripts` folder. 

## Pre-processing bulk images for machine learning 
The `image_preprocessing` folder contains utility scripts to:
1. Download the bulk images `download_data.py`
2. Assemble the annotated patches into tiles `assemble_annotations.py`
3. Crop the bulk images `crop_bulk_imgs.py`
4. Divide the bulk images into tiles `tile_imgs.py`
5. And lastly, postprocess the sliced annotations `postprocess_dataset.py`
The entire preprocessing pipeline can be run via 
```bash
sbatch pipeline.sh
``` 

## Training and Inference
The `CutLER`, `Mask-RCNN`, `Mask2Former`, `MaskDINO`, and `Grounded-SAM-2` submodules each have their own usage instructions (see the `MassID45 Instructions` section in each submodule's `README.md`). We ran our experiments using 4 RTX6000 GPUs for training, and 1 RTX6000 GPU for testing. 

## Model Checkpoints
Model checkpoints can be downloaded at [Zenodo](https://zenodo.org/records/15479862).




