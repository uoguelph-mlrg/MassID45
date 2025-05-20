# MassID45
This repo contains code for the MassID45 dataset: a multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level.

# Installation
Dependencies for downloading and preprocessing the bulk image data can be installed via:
```bash
pip install -r requirements.txt
```
The `CutLER`, `Mask-RCNN`, `Mask2Former`, `MaskDINO`, and `Grounded-SAM-2` submodules each contain their own installation and usage instructions (see the `MassID45 Instructions` section in each `README`). We recommend creating separate Python or Conda environments for each submodule.

# Usage

## Downloading Image Data

## Annotation utilities
Code to run the watershed algorithm and crop the bulk images into annotator patches is contained in the `arth-imrec` submodule.

## Pre-processing bulk images for machine learning 





