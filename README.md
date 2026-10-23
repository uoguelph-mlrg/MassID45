# MassID45
This repo contains the preprocessing and training/inference code for MassID45 (Mixed Arthropod Sample Segmentation and Identification), a multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level.

Read our paper in [*Scientific Data*](https://doi.org/10.1038/s41597-026-07251-x) (Orsholm, Quinto et al., 2026). If you use MassID45 in your research, please consider [citing us](#citation).

# Data
- Images, annotations and model checkpoints: [Zenodo](https://doi.org/10.5281/zenodo.18963816)
- Bulk-sample metabarcoding reads: [ENA project PRJEB86111](https://www.ebi.ac.uk/ena/browser/view/PRJEB86111)
- Individual-specimen DNA barcodes: [BOLD dataset DS-LPEPA22](https://doi.org/10.5883/DS-LPEPA22)

# Installation
Requirements for downloading and preprocessing the data can be found in the `image_preprocessing` folder. We recommend creating a dedicated Python environment for `image_preprocessing`.

The `CutLER`, `Mask-RCNN`, `Mask2Former`, `MaskDINO`, and `Grounded-SAM-2` submodules each contain their own installation instructions (see the `INSTALL.md` files in each submodule). We recommend creating separate Python or Conda environments for each submodule.

# Usage

## Annotation utilities
Code to run the watershed algorithm and crop the bulk images into annotator patches is contained in the `arth-imrec` submodule.

## Downloading bulk imagery data
The bulk images and annotation tiles can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.18963816) by running `image_preprocessing/download_data.py`. Scripts for downloading the ENA sequence data can be found in the `image_preprocessing/ENA_sequence_scripts` folder.

## Pre-processing bulk images for machine learning 
The `image_preprocessing` folder contains utility scripts for downloading and assembling the bulk image data. The data pipeline is as follows:
1. Download the bulk images `download_data.py`
2. Assemble the annotated patches into the full bulk images `assemble_annotations.py`
3. Crop the bulk images to exclude excess areas `crop_bulk_imgs.py`
4. Divide the bulk images into tiles for training `tile_imgs.py`
5. And lastly, postprocess the sliced annotations `postprocess_dataset.py`
The entire preprocessing pipeline can be run via 
```bash
sbatch pipeline.sh
``` 

## Training and Inference
The `CutLER`, `Mask-RCNN`, `Mask2Former`, `MaskDINO`, and `Grounded-SAM-2` submodules each have their own usage instructions (see the `MassID45 Instructions` section in each submodule's `README.md`). We ran our experiments using 4 RTX6000 GPUs for training, and 1 RTX6000 GPU for testing. 

## Model Checkpoints
Model checkpoints (`model_checkpoints.zip`) can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.18963816).

# Citation
If you find MassID45 useful in your research, please consider citing:

```bibtex
@article{orsholm2026massid45,
  author={Orsholm, Johanna and Quinto, John and Autto, Hannu and Banelyte, Gaia
    and Chazot, Nicolas and deWaard, Jeremy and deWaard, Stephanie
    and Farrell, Arielle and Furneaux, Brendan and Hardwick, Bess and Ito, Nao
    and Kar, Amlan and Kalttop{\"a}{\"a}, Oula and Kerdraon, Deirdre
    and Kristensen, Erik and McKeown, Jaclyn and Mononen, Tommi and Nein, Ellen
    and Rogers, Hanna and Roslin, Tomas and Schmitz, Paula and Sones, Jayme
    and Sujala, Maija and Thompson, Amy and Zakharov, Evgeny V.
    and Zarubiieva, Iuliia and Gupta, Akshita and Lowe, Scott C.
    and Taylor, Graham W.},
  title={A multi-modal dataset for insect biodiversity with imagery and {DNA}
    at the trap and individual level},
  journal={Scientific Data},
  volume={13},
  pages={630},
  year={2026},
  month=apr,
  doi={10.1038/s41597-026-07251-x},
}
```

An earlier version of this work appeared at the NeurIPS 2025 Workshop for Imageomics ([OpenReview](https://openreview.net/forum?id=Kn8bz1ZPvE), [arXiv](https://arxiv.org/abs/2507.06972)). Tiny instance segmentation on MassID45 is studied further in BugSR ([OpenReview](https://openreview.net/forum?id=OYK7psfeNe)).
