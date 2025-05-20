
import json
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from sahi.slicing import slice_coco
import matplotlib.pyplot as plt

def check_multipolygons(coco_data):
    # Check that assembled LIFEPLAN data doesn't have any multipolygon annotations
    mp_annots = []
    for annot in coco_data['annotations']:
        if len(annot['segmentation']) > 1:
            mp_annots.append(annot['segmentation'])
    assert len(mp_annots) == 0, "Annotations with multipolygons found"

def check_negative_coords(coco_data):
   # Check if any annotation segmentation entries within assembled_coco_data have negative coordinates
    has_negative_coords = False
    for annotation in coco_data['annotations']:
        for segment in annotation['segmentation']:
            if any(coord < 0 for coord in segment):
                has_negative_coords = True
                break
        if has_negative_coords:
            break
    assert has_negative_coords == False
    print(f"Has negative coordinates: {has_negative_coords}")
   
SR_METHOD = "swinir_bioscan" # Change as needed
base_dir = f"/scratch/ssd004/scratch/jquinto/lifeplan_b_v9_cropped_center_sr_{SR_METHOD}" 

# Iterate through each train2017 and val2017 JSON
with open(f"{base_dir}/annotations/instances_train2017.json", "r") as file:
    train_data = json.load(file)
with open(f"{base_dir}/annotations/instances_val2017.json", "r") as file:
    val_data = json.load(file)
with open(f"{base_dir}/annotations/instances_test2017.json", "r") as file:
    test_data = json.load(file)

# Sanity checks for dataset
for dataset in [train_data, val_data, test_data]:
   check_multipolygons(dataset)
   check_negative_coords(dataset)

def get_sahi_coco_data(crop_fac, crop_rows, 
                       crop_cols, split, overlap_ratio = 0.4, 
                       min_area_ratio = 0.1, sahi_file_path = "/h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets"):
  # Setup COCO dataset file structure
  sahi_file_path = f"{sahi_file_path}/sahi_{crop_fac}_ignore_neg_SR"
  
  if os.path.exists(sahi_file_path) is False:
    os.makedirs(sahi_file_path)
  if os.path.exists(f"{sahi_file_path}/{split}2017") is False:
    os.makedirs(f"{sahi_file_path}/{split}2017")
  if os.path.exists(f"{sahi_file_path}/annotations") is False:
    os.makedirs(f"{sahi_file_path}/annotations")

  # Perform slicing
  coco_dict, coco_path = slice_coco(
      coco_annotation_file_path=f"{base_dir}/annotations/instances_{split}2017.json",
      image_dir=f"{base_dir}/{split}2017/",
      output_coco_annotation_file_name=f"instances_{split}2017",                                                                                                
      ignore_negative_samples=False,
      output_dir=f"{sahi_file_path}/{split}2017_initial",
      slice_height=crop_rows,
      slice_width=crop_cols,
      overlap_height_ratio=overlap_ratio,
      overlap_width_ratio=overlap_ratio,
      min_area_ratio=min_area_ratio, # if ratio of cut-off annotation/original annotation < threshold, ignore it
      verbose=True
  )

  sliced_json_path = f"{sahi_file_path}/{split}2017_initial/instances_{split}2017_coco.json"
  with open(sliced_json_path, "r") as f:
    sliced_coco_data = json.load(f)

  # Copy valid images (non-empty slices) to its own folder
  valid_imgs = [json_entry['file_name'] for json_entry in sliced_coco_data['images']]

  # Create image directory to store non-empty images
  no_empty_path = f"{sahi_file_path}/{split}2017"
  if os.path.exists(no_empty_path) is False:
    os.makedirs(no_empty_path)

  # Create archive directory to clean up files and folders
  archive_path = f"{sahi_file_path}/archive"
  if os.path.exists(archive_path) is False:
    os.makedirs(archive_path)

   # Move sliced images to non-empty image directory
  for img in tqdm(sliced_coco_data["images"]):
    img_path = img["file_name"]
    if img_path in valid_imgs:
        source = f"/{sahi_file_path}/{split}2017_initial/{img_path}"
        dest = f"{no_empty_path}/{img_path}"
        shutil.copyfile(source, dest)

  # Move COCO json file to the proper 'annotations' folder
  shutil.move(sliced_json_path, f"{sahi_file_path}/annotations/instances_{split}2017.json")

  # Clean up
  shutil.move(f"{sahi_file_path}/{split}2017_initial", f"{archive_path}/{split}2017_initial")

  return sliced_coco_data

# SR upscaling factor of 2 will be used, based on past experiments
ZOOM_FACTOR = 2
CROP_SZ = 1024
print(f"Zoom Factor: {ZOOM_FACTOR} | Crop Size: {CROP_SZ} | SR Method: {SR_METHOD}")
SAHI_FILE_PATH = f"/h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_{SR_METHOD}" 

sliced_coco_data_train = get_sahi_coco_data(
    crop_fac = CROP_SZ,
    crop_rows = CROP_SZ,
    crop_cols = CROP_SZ,
    split = "train",
    overlap_ratio = 0.6,
    min_area_ratio = 0.1,
    sahi_file_path = SAHI_FILE_PATH
)
sliced_coco_data_val = get_sahi_coco_data(
    crop_fac = CROP_SZ,
    crop_rows = CROP_SZ,
    crop_cols = CROP_SZ,
    split = "val",
    overlap_ratio = 0.6,
    min_area_ratio = 0.1,
    sahi_file_path = SAHI_FILE_PATH
)
sliced_coco_data_train = get_sahi_coco_data(
    crop_fac = CROP_SZ,
    crop_rows = CROP_SZ,
    crop_cols = CROP_SZ,
    split = "test",
    overlap_ratio = 0.6,
    min_area_ratio = 0.1,
    sahi_file_path = SAHI_FILE_PATH
)