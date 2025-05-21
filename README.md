# MassID45
This repo contains code for the MassID45 dataset: a multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level.

# Installation
The `CutLER`, `Mask-RCNN`, `Mask2Former`, `MaskDINO`, and `Grounded-SAM-2` submodules each contain their own installation instructions (see the `INSTALL.md` files in each submodule). We recommend creating separate Python or Conda environments for each submodule.

# Usage

## Annotation utilities
Code to run the watershed algorithm and crop the bulk images into annotator patches is contained in the `arth-imrec` submodule.

## Downloading bulk imagery data
The bulk images can be downloaded from Google Drive by running `image_preprocessing/download_data.py`.

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

Training scripts are based on Detectron2, and are typically structured as follows:
```bash
usage: train_net.py [-h] [--config-file FILE] [--resume] [--eval-only] [--num-gpus NUM_GPUS]
                    [--num-machines NUM_MACHINES] [--machine-rank MACHINE_RANK] [--dist-url DIST_URL] [--eval_only]
                    [--EVAL_FLAG EVAL_FLAG] [--dataset_path DATASET_PATH] [--exp_id EXP_ID]
                    ...

positional arguments:
  opts                  Modify config options at the end of the command. For Yacs configs, use space-separated
                        "PATH.KEY VALUE" pairs. For python-based LazyConfig, use "path.key=value".

optional arguments:
  -h, --help            show this help message and exit
  --config-file FILE    path to config file
  --resume              Whether to attempt to resume from the checkpoint directory. See documentation of
                        `DefaultTrainer.resume_or_load()` for what it means.
  --eval-only           perform evaluation only
  --num-gpus NUM_GPUS   number of gpus *per machine*
  --num-machines NUM_MACHINES
                        total number of machines
  --machine-rank MACHINE_RANK
                        the rank of this machine (unique per machine)
  --dist-url DIST_URL   initialization URL for pytorch distributed backend. See
                        https://pytorch.org/docs/stable/distributed.html for details.
  --eval_only
  --EVAL_FLAG EVAL_FLAG
  --dataset_path DATASET_PATH
                        Path to the dataset directory containing annotations and images
  --exp_id EXP_ID     Identifier string -- tile size for training model 

Examples:

Run on single machine:
    $ train_net.py --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ train_net.py --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ train_net.py --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ train_net.py --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
```
Inference scripts are based on the Slicing-Aided Hyper-Inference (SAHI) library, and are typically structured as follows:
```bash
usage: sahi_inference.py [-h] [--model_path MODEL_PATH] [--exp_name EXP_NAME] [--predict]
                         [--dataset_json_path DATASET_JSON_PATH] [--dataset_img_path DATASET_IMG_PATH]
                         [--config_path CONFIG_PATH] [--crop_fac CROP_FAC]
                         [--postprocess_match_threshold POSTPROCESS_MATCH_THRESHOLD]
                         [--model_confidence_threshold MODEL_CONFIDENCE_THRESHOLD] [--scale_factor SCALE_FACTOR]
                         [--super_resolution] [--slice_height SLICE_HEIGHT] [--slice_width SLICE_WIDTH]
                         [--image_size IMAGE_SIZE] [--overlap OVERLAP]

Run SAHI prediction and COCO evaluation.

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the model file.
  --exp_name EXP_NAME   Experiment Name
  --predict
  --dataset_json_path DATASET_JSON_PATH
                        Path to the dataset JSON file.
  --dataset_img_path DATASET_IMG_PATH
                        Path to the dataset image file(s).
  --config_path CONFIG_PATH
                        Path to the config file.
  --crop_fac CROP_FAC   Crop factor for slicing the image.
  --postprocess_match_threshold POSTPROCESS_MATCH_THRESHOLD
                        Post-process match threshold.
  --model_confidence_threshold MODEL_CONFIDENCE_THRESHOLD
                        Model Confidence Threshold.
  --scale_factor SCALE_FACTOR 
                        Scale factor when super resolution is applied during inference (coming soon)
  --super_resolution
                        Indicates whether images and tiles have super resolution applied to them (coming soon)
  --slice_height SLICE_HEIGHT
                        Height of tiles, in pixels (default: 512)
  --slice_width SLICE_WIDTH
                        Width of tiles, in pixels (default: 512)
  --image_size IMAGE_SIZE
                        Input image size for model (default: 1024)
  --overlap OVERLAP
                        Overlap fraction between tiles (0-1)
```




