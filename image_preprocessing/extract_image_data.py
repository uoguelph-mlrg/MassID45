import torch
import gdown
import zipfile

import json
import os, sys
import cv2
import numpy as np
import glob
import shutil
from collections import Counter
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import copy
import pickle
from shapely.geometry import Point, Polygon, MultiPolygon, box, GeometryCollection
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm


# Replace with the correct file path
file_path = "/content/drive/MyDrive/annots_20250307_coco.json"

# Read the file and load JSON data
with open(file_path, "r") as file:
  coco_format = json.load(file)

# Trim to bugs only
coco_format['annotations'] = [annot for annot in coco_format['annotations'] if annot['name'] == 'b']

for img in coco_format['images']:
  try:
    img['path'] = img['toras_path']
    img['height'] = cv2.imread(f"/content/{img['path']}").shape[0]
    img['width'] = cv2.imread(f"/content/{img['path']}").shape[1]
    img['file_name'] = img['toras_path'].split('/')[-1]
  except:
    try:
      img['toras_path'] = img['toras_path'].replace('batch-1', 'batch-2')
      img['path'] = img['toras_path']
      img['height'] = cv2.imread(f"/content/{img['path']}").shape[0]
      img['width'] = cv2.imread(f"/content/{img['path']}").shape[1]
      img['file_name'] = img['toras_path'].split('/')[-1]
    except:
      print(f"Image not found: {img['toras_path']}")

unique_bulk_imgs = set([im["file_name"][:-9] for im in coco_format['images']])
print(len(unique_bulk_imgs))

merged_polys, merged_multi_polys = [], []
for m_annot in coco_format['annotations']:
    polygons = []
    for polygon_idx in range(len(m_annot['segmentation'])):
        coco_coords = m_annot['segmentation'][polygon_idx]
        segment_coords = np.array(coco_coords).reshape(-1, 2)
        if len(segment_coords) >= 3:
            polygons.append(make_valid(Polygon(segment_coords)))
        else:
            continue
    if len(polygons) > 0:
        union_result = unary_union(polygons)

        # Handle different geometry types
        if isinstance(union_result, GeometryCollection):
            # Extract only polygon geometries from collection
            poly_geoms = [geom for geom in union_result.geoms
                        if isinstance(geom, (Polygon, MultiPolygon))]
            if poly_geoms:
                # Find the largest polygon
                largest_poly = max(poly_geoms, key=lambda x: x.area)
                if isinstance(largest_poly, MultiPolygon):
                    largest_single_poly = max(largest_poly.geoms, key=lambda x: x.area)
                    merged_multi_polys.append(largest_single_poly)
                    merged_polys.append(largest_single_poly)
                    m_annot['segmentation'] = [[list(i) for i in list(zip(*largest_single_poly.exterior.coords.xy))]]
                else:  # Single Polygon
                    merged_polys.append(largest_poly)
                    m_annot['segmentation'] = [[list(i) for i in list(zip(*largest_poly.exterior.coords.xy))]]
            else:
                # If no polygon geometries found, remove annotation
                coco_format['annotations'].remove(m_annot)
                continue

        elif isinstance(union_result, MultiPolygon):
            merged_multi_polys.append(union_result)
            # print(union_result)
            # Get the largest polygon from the MultiPolygon
            largest_poly = max(union_result.geoms, key=lambda x: x.area)
            merged_polys.append(largest_poly)
            m_annot['segmentation'] = [[list(i) for i in list(zip(*largest_poly.exterior.coords.xy))]]

        elif isinstance(union_result, Polygon):
            merged_polys.append(union_result)
            m_annot['segmentation'] = [[list(i) for i in list(zip(*union_result.exterior.coords.xy))]]

        else:  # LineString, Point, or other geometry types
            coco_format['annotations'].remove(m_annot)
            continue
    else:
        coco_format['annotations'].remove(m_annot)

print(f"Number of merged polygons: {len(merged_polys)}")

def shift_annotations(coco_data, metadata, img_ids, tile_idxs, min_size = 15):
  # Shift annotations
  shifted_annotations, invalid_polygons = [], []
  invalid_poly_count, total = 0, 0
  for img_id, tile_idx in zip(img_ids, tile_idxs):
    # Get all annotations corresponding to a particular tile
    annotations = copy.deepcopy([i for i in coco_data['annotations'] if i['image_id'] == img_id])

    # Obtain coordinate limits from metadata
    metadata_y_shift = metadata[tile_idx][0]
    metadata_x_shift = metadata[tile_idx][2]
    # print(img_id, tile_idx, f"{[metadata_x_shift, metadata_y_shift]}")

    # Iterate through annotations
    for annot in annotations:
      revised_coords, polygons = [], []
      total += len(annot['segmentation'])
      for polygon_idx in range(len(annot['segmentation'])):
        segment_coords = np.array(annot['segmentation'][polygon_idx])
        segment_coords[:, 0] += metadata_x_shift
        segment_coords[:, 1] += metadata_y_shift
        # First check if the polygon has 3 or more points; invalid otherwise
        if segment_coords.shape[0] >= 3:
          poly = Polygon(segment_coords)
          if poly.is_valid:
            revised_coords.append(segment_coords.tolist())
            polygons.append(poly)
          else:
              invalid_poly_count += 1
              invalid_polygons.append(segment_coords.tolist())  # Add invalid polygon
        # else:
        #   invalid_poly_count += 1

      if len(polygons) > 0:
        ### [x1, y1]
        ### [[x1, ,y1], ]
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        annot['area'] = area
        annot['bbox'] = bbox
        revised_coords = [np.array(coord).ravel().tolist() for coord in revised_coords] ### CONVERTS TO DESIRED COCO (XY) FORMAT
        annot['segmentation'] = revised_coords
        # Only add if area >= 15 pixels square
        if area >= min_size:
          shifted_annotations.append(annot)
      else:
        annot['area'] = 0
        annot['bbox'] = [0, 0, 0, 0]

  print(f"Invalid polygons: {invalid_poly_count} | Total: {total}")

  return shifted_annotations, invalid_polygons

def draw_mask(image, mask_data, color, shape='polygon'):
    mask = np.zeros_like(image)
    if shape == 'square':
        # Draw a square using the bounding box coordinates
        x, y, width, height = cv2.boundingRect(np.array(mask_data, dtype=np.int32))
        cv2.rectangle(mask, (x, y), (x + width, y + height), color, -1)
    else:
        # Original polygon fill logic
        segments = [
            np.array(segment, dtype=np.int32).reshape((-1, 1, 2)) if np.array(segment, dtype=np.int32).shape[0] == 1 else np.array(segment, dtype=np.int32).reshape((1, -1, 2)) for segment in mask_data
        ]
        cv2.fillPoly(mask, segments, color)
    return cv2.addWeighted(image, 1, mask, 0.5, 0)

def plot_shifted_annotations(annotations, invalid_polygons, img_name, save_img, plot_annotations, save_path):
    if plot_annotations is True:
        image = cv2.imread(f"/content/bulk_batch_1_and_2/{img_name}/{img_name}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw each valid mask
        for ann in annotations:
            segmentation = [np.array(seg).reshape(-1, 2).flatten().tolist() for seg in ann['segmentation']]
            if ann['name'] == "b":  # GREEN = BUG
                color = (0, 255, 0)
            elif ann["name"] == "d":  # PURPLE = DEBRIS
                color = (255, 0, 143)
            elif ann["name"] == "u":  # ORANGE = UNKNOWN or EDGE
                color = (0, 165, 255)
            elif ann["name"] == "e":  # RED = ENTITY
                color = (0, 0, 255)
            image = draw_mask(image, segmentation, color)

        # Draw invalid masks as squares
        for invalid_polygon in invalid_polygons:
            color = (255, 0, 0)  # Red color for invalid polygons
            # Call draw_mask with shape='square' to draw a bounding box square
            image = draw_mask(image, invalid_polygon, color, shape='square')

        # Display and save logic remains the same
        plt.imshow(image)
        plt.title(img_name)
        plt.axis('off')
        plt.show()

    if save_img is True:
        cv2.imwrite(f"{save_path}/{img_name}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def project_tiled_annots(metadata_path, coco_data, save_img = False, plot_annotations = True, save_path = None):
  # coco_data = copy.deepcopy(coco_format)
  # Load metadata npy file
  metadata = np.load(metadata_path)

  # First get all image tiles belonging to a particular image
  img_name = metadata_path.split('/')[3]
  tile_imgs = [i for i in coco_data['images'] if img_name in i['path']]

  if len(tile_imgs) == 0:
    print("No annotated images found")
    return [], []

  # Now get corresponding IDs, file names, and tile indices
  img_ids = [entry['id'] for entry in tile_imgs]
  img_names = [entry['file_name'] for entry in tile_imgs]

  # Assume row of tile coordinates is its index
  tile_idxs = [int(file_name.split('_')[-1].split('.')[0]) for file_name in img_names]

  # Shift annotations
  shifted_annotations, invalid_polygons = shift_annotations(coco_data, metadata, img_ids, tile_idxs)
  for annot in shifted_annotations:
    annot['image_id'] = hash(img_name)

  # Visualize
  # plot_shifted_annotations(shifted_annotations, img_name, save_img, plot_annotations, save_path)
  plot_shifted_annotations(shifted_annotations, invalid_polygons, img_name, save_img, plot_annotations, save_path)

  return shifted_annotations, invalid_polygons

def get_img_names(directory):
  img_names = glob.glob(f"{directory}/*")
  img_names = [folder.split('/')[-1] for folder in img_names]

  return img_names

bulk_imgs = get_img_names('/content/bulk_batch_1_and_2')
batch_1_imgs = get_img_names('/content/batch-1')
batch_2_imgs = get_img_names('/content/batch-2')

print(len(bulk_imgs), len(batch_1_imgs), len(batch_2_imgs))

SAVE_PATH = "/content/drive/MyDrive/LIFEPLAN_Annotated_Bulk_Imgs/Revised_Images_b_Mar_2025"
if os.path.exists(SAVE_PATH) is False:
  os.makedirs(SAVE_PATH)

# Not started or ongoing as of **February 7**
EXCLUSION_LIST = []

batch_1_annotations = []
batch_1_invalid_polygons = []
for img in batch_1_imgs:
  if img in EXCLUSION_LIST:
    continue
  metadata_path = f"/content/batch-1/{img}/{img}_tile_metadata.npy"
  print(img)
  shifted_annotations, invalid_polygons = project_tiled_annots(
      metadata_path,
      coco_format,
      save_img = False,
      plot_annotations = False,
      save_path = SAVE_PATH
  )
  batch_1_annotations.extend(shifted_annotations)
  batch_1_invalid_polygons.append({
      "img": img,
      "invalid_polygons": invalid_polygons,
      "n_invalid_polygons": len(invalid_polygons)
  })

batch_2_annotations = []
batch_2_invalid_polygons = []
for img in batch_2_imgs:
  if img in EXCLUSION_LIST:
    continue
  metadata_path = f"/content/batch-2/{img}/{img}_tile_metadata.npy"
  print(img)
  shifted_annotations, invalid_polygons = project_tiled_annots(
      metadata_path,
      coco_format,
      save_img = False,
      plot_annotations = False,
      save_path = SAVE_PATH
  )
  batch_2_annotations.extend(shifted_annotations)
  batch_2_invalid_polygons.append({
      "img": img,
      "invalid_polygons": invalid_polygons,
      "n_invalid_polygons": len(invalid_polygons)
  })

full_batch_annotations = batch_1_annotations + batch_2_annotations
for annot in full_batch_annotations:
  annot['iscrowd'] = 0

print(len(full_batch_annotations))

# Get available images
annotated_imgs = glob.glob(f"{SAVE_PATH}/*.png")
annotated_img_names = [img.split('/')[-1].split('.')[0] for img in annotated_imgs]
print(len(annotated_imgs))

assert len(annotated_imgs) == 49 , print("Number of annotated bulk images is incorrect; check exclusion list?")

# Get image property of COCO JSON
lifeplan_coco_imgs = []
for img in annotated_imgs:
  img_path = img.split('/')[-1]
  img_name = img_path.split('.')[0]
  img_dict = {}
  img_dict['id'] = hash(img_name)
  img_dict['file_name'] = f"{img_name}.jpg"

  loaded_img = cv2.imread(img)
  height, width = loaded_img.shape[0], loaded_img.shape[1]
  img_dict['height'] = height
  img_dict['width'] = width
  img_dict["path"] = f"bulk_batch_1_and_2/{img_name}/{img_name}.jpg"

  lifeplan_coco_imgs.append(img_dict)

hashed_names = []
for i in lifeplan_coco_imgs:
  hashed_names.append(i['id'])

image_ids = []
for i in full_batch_annotations:
  image_ids.append(i['image_id'])

assert set(hashed_names) == set(image_ids), print("Hashed image IDs do not match")

assembled_coco_data = {}
assembled_coco_data['info'] = {}
assembled_coco_data['licenses'] = []
assembled_coco_data['categories'] = coco_format['categories']
assembled_coco_data['images'] = lifeplan_coco_imgs
assembled_coco_data['annotations'] = full_batch_annotations

# First create folder structure as expected by MaskDINO
base_dir = "/content/drive/MyDrive/lifeplan_b_v9"
# ! rm -r /content/drive/MyDrive/lifeplan_category_name_v2
if os.path.exists(base_dir) is False:
    os.mkdir(base_dir)
for folder in ["annotations", "train2017", "val2017", "test2017"]:
    folder_path = f"{base_dir}/{folder}"
    if os.path.exists(folder_path) is False:
        os.mkdir(folder_path)

from sklearn.model_selection import train_test_split

# Keep consistent w/ past experiments
VAL_PREFIXES = ['GYTL2T', 'GUSA7Z', 'GRLLN6']
TEST_PREFIXES = ['G9XHGJ', 'G5JHXH', 'GGCNMZ', 'GZDUDR', 'GEB17N', 'GPRD1F']
# TEST_PREFIXES = ['G9XHGJ', 'GP8K9U', 'G5JHXH', 'GGCNMZ', 'GZDUDR', 'GEB17N', 'GPRD1F']

def split_coco_dataset(coco_json_path, train_fraction, valid_fraction, test_fraction):
    # Load the original COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Assuming coco_data has 'images' and 'annotations' keys
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Group images by their prefix
    image_groups = {}
    for image in images:
        prefix = image['file_name'][:-4]
        if prefix not in image_groups:
            image_groups[prefix] = []
        image_groups[prefix].append(image)

    # Split the prefixes into train, valid, and test groups
    prefixes = list(image_groups.keys())
    train_prefixes = set(prefixes) - set(VAL_PREFIXES) - set(TEST_PREFIXES)
    valid_prefixes = set(VAL_PREFIXES)
    test_prefixes = set(TEST_PREFIXES)

    # train_prefixes, test_prefixes = train_test_split(prefixes, test_size=test_fraction, random_state=42)
    # train_prefixes, valid_prefixes = train_test_split(train_prefixes, test_size=valid_fraction/train_fraction, random_state=42)

    print(len(train_prefixes)/len(prefixes), len(valid_prefixes)/len(prefixes), len(test_prefixes)/len(prefixes), len(train_prefixes)/len(prefixes) + len(valid_prefixes)/len(prefixes) + len(test_prefixes)/len(prefixes))

    # Function to collect images and annotations based on prefixes
    def collect_data(prefixes):
        images = []
        annotations = []
        for prefix in prefixes:
            images.extend(image_groups[prefix])
            # Collect annotations for the selected images
            image_ids = [image['id'] for image in image_groups[prefix]]
            for annotation in coco_data['annotations']:
                if annotation['image_id'] in image_ids:
                    annotations.append(annotation)
        return images, annotations

    # Collect data for each set
    train_images, train_annotations = collect_data(train_prefixes)
    valid_images, valid_annotations = collect_data(valid_prefixes)
    test_images, test_annotations = collect_data(test_prefixes)

    # Function to create a new COCO dataset
    def create_coco_dataset(images, annotations, output_path):
        new_coco_dataset = {
            # "info": coco_data["info"],
            # "licenses": coco_data["licenses"],
            "images": images,
            "annotations": annotations,
            "categories": coco_data['categories']
        }
        with open(output_path, 'w') as f:
            json.dump(new_coco_dataset, f, indent=4)

    # Create new COCO datasets
    create_coco_dataset(train_images, train_annotations, f"{base_dir}/annotations/instances_train2017.json")
    create_coco_dataset(valid_images, valid_annotations, f"{base_dir}/annotations/instances_val2017.json")
    create_coco_dataset(test_images, test_annotations, f"{base_dir}/annotations//instances_test2017.json")

# Example usage
split_coco_dataset(f"{SAVE_PATH}/coco_format_assembled_b_v9.json", 0.8, 0.05, 0.15)

# Iterate through each train2017 and val2017 JSON
with open(f"{base_dir}/annotations/instances_train2017.json", "r") as file:
    train_data = json.load(file)
with open(f"{base_dir}/annotations/instances_val2017.json", "r") as file:
    val_data = json.load(file)
with open(f"{base_dir}/annotations/instances_test2017.json", "r") as file:
    test_data = json.load(file)

def remove_empty_annotations(data):
  removed_ctr = 0
  # Remove empty annotations
  for annotation in data['annotations']:
    if 'segmentation' not in annotation.keys() or len(annotation['segmentation']) == 0:
        data['annotations'] = [item for item in data['annotations'] if item != annotation]
        removed_ctr += 1
  print(f"Number of empty annotations removed: {removed_ctr}")

  return data

def get_empty_annotations(data):
  empty_annots = []
  # Remove empty annotations
  for annotation in data['annotations']:
    if 'segmentation' not in annotation.keys() or len(annotation['segmentation']) == 0:
      empty_annots.append(annotation)
  print(f"Number of empty annotations: {len(empty_annots)}")

  return empty_annots

train_data = remove_empty_annotations(train_data)
val_data = remove_empty_annotations(val_data)
test_data = remove_empty_annotations(test_data)

train_empty_annots = get_empty_annotations(train_data)
val_empty_annots = get_empty_annotations(val_data)
test_empty_annots = get_empty_annotations(test_data)


# Generate annotation files with taxonomic labels
with open(f"{base_dir}/annotations/instances_train2017_tax.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open(f"{base_dir}/annotations/instances_val2017_tax.json", "w") as f:
    json.dump(val_data, f, indent=4)
with open(f"{base_dir}/annotations/instances_test2017_tax.json", "w") as f:
    json.dump(test_data, f, indent=4)

def generate_agnostic_bug_labels(coco_data):
  agnostic_coco_data = copy.deepcopy(coco_data)
  agnostic_coco_data['categories'] = [{
      "id": 1,
      "name": "b",
      "supercategory": "insect"
  }]
  for annot in agnostic_coco_data['annotations']:
    annot['category_id'] = 1

  return agnostic_coco_data

# Annotation files WITHOUT taxonomic labels
agnostic_train_data = generate_agnostic_bug_labels(train_data)
agnostic_val_data = generate_agnostic_bug_labels(val_data)
agnostic_test_data = generate_agnostic_bug_labels(test_data)

with open(f"{base_dir}/annotations/instances_train2017.json", "w") as f:
    json.dump(agnostic_train_data, f, indent=4)
with open(f"{base_dir}/annotations/instances_val2017.json", "w") as f:
    json.dump(agnostic_val_data, f, indent=4)
with open(f"{base_dir}/annotations/instances_test2017.json", "w") as f:
    json.dump(agnostic_test_data, f, indent=4)

def copy_images_to_coco_folder(data, split):
    for img in data["images"]:
        img_path = img["path"]
        source = f"/content/{img_path}"
        dest = f"{base_dir}/{split}2017/{img['file_name']}"
        shutil.copyfile(source, dest)
        print(source, dest)

copy_images_to_coco_folder(train_data, "train")
copy_images_to_coco_folder(val_data, "val")
copy_images_to_coco_folder(test_data, "test")