
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Point, Polygon, MultiPolygon, box, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import glob
import pickle
import argparse
import shutil

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
   
def check_crop_szs(coco_data, crop_sz):
    diff_crop_sz = []
    for img in coco_data['images']:
        if int(img['height']) != crop_sz or int(img['width']) != crop_sz:
            diff_crop_sz.append(img)
    assert len(diff_crop_sz) == 0

def get_sahi_tiled_annots(sliced_coco_data, min_size=15, remove_cutoff=False):
    """
    Applies masking to invalid sliced segmentation masks
    """
    tiled_annotations, invalid_annotations, too_small_annotations, cutoff_annotations = {}, {}, {}, {}
    total, invalid_annot_count, too_small_annot_count, cutoff_count = 0, 0, 0, 0

    # Get all tiled images from JSON file
    tiled_images = sliced_coco_data['images']

    for tiled_img in tiled_images:
        # Initialize lists for this image
        valid_tiled_annots = []
        invalid_polygons_tile = []
        too_small_polygons_tile = []
        cutoff_polygons_tile = []
        
        # Get image properties
        img_name = tiled_img['file_name']
        image_width = tiled_img['width']
        image_height = tiled_img['height']
        
        # Filter annotations to those in the target tiled image
        desired_annots = [annot for annot in sliced_coco_data['annotations'] 
                         if annot['image_id'] == tiled_img['id']]
        
        # Iterate through these annotations
        for annot in desired_annots:
            revised_coords = []
            polygons = []
            
            for polygon_idx in range(len(annot['segmentation'])):
                segment_coords = np.array(annot['segmentation'][polygon_idx]).reshape(-1, 2)
                # First check if the polygon has 3 or more points; invalid otherwise
                if segment_coords.shape[0] >= 3:
                    poly = Polygon(segment_coords)
                    # Check for validity
                    if poly.is_valid:
                        # Now check if it is "cut-off" between tiles
                        is_cutoff = False
                        for x, y in poly.exterior.coords:
                            if x == 0 or x == image_width or y == 0 or y == image_height:
                                is_cutoff = True
                                cutoff_polygons_tile.append(segment_coords.tolist())
                                cutoff_count += 1
                                break
                        if remove_cutoff is True:
                          if not is_cutoff:
                            revised_coords.append(segment_coords.tolist())
                            polygons.append(poly)
                        else:
                          revised_coords.append(segment_coords.tolist())
                          polygons.append(poly)
                    else:
                        invalid_annot_count += 1
                        invalid_polygons_tile.append(segment_coords.tolist())

            if len(polygons) > 0:
                multi_poly = MultiPolygon(polygons)
                x, y, max_x, max_y = multi_poly.bounds
                width = max_x - x
                height = max_y - y
                bbox = (x, y, width, height)
                area = multi_poly.area

                annot_copy = annot.copy()  # Create a copy to avoid modifying original
                annot_copy['area'] = area
                annot_copy['bbox'] = bbox
                revised_coords = [np.array(coord).ravel().tolist() for coord in revised_coords] ### CONVERTS TO DESIRED COCO (XY) FORMAT
                annot_copy['segmentation'] = revised_coords

                if area >= min_size:
                    valid_tiled_annots.append(annot_copy)
                else:
                    too_small_polygons_tile.append(revised_coords)
                    too_small_annot_count += 1

        # After processing all annotations for this image, update the dictionaries
        # if valid_tiled_annots:  # Only add if there are valid annotations
        tiled_annotations[img_name[:-4]] = valid_tiled_annots
        # if invalid_polygons_tile:
        invalid_annotations[img_name[:-4]] = invalid_polygons_tile
        # if too_small_polygons_tile:
        too_small_annotations[img_name[:-4]] = too_small_polygons_tile
        # if cutoff_polygons_tile:
        cutoff_annotations[img_name[:-4]] = cutoff_polygons_tile
        
        total += len(valid_tiled_annots)

    return tiled_annotations, invalid_annotations, too_small_annotations, cutoff_annotations

# Load test image for masking out invalid or cut-off images (if necessary or if remove_cutoff is True)
test_img = cv2.imread("assets/G1BMRA_0010.jpg", cv2.IMREAD_COLOR)
background_value = test_img[1400:1450, 2000:2050, :]

def draw_mask(image, mask_data, color, shape='polygon', roi = None):
    mask = np.zeros_like(image)
    if shape == 'square':
        # Draw a square using the bounding box coordinates
        x, y, width, height = cv2.boundingRect(np.array(mask_data, dtype=np.int32))
        cv2.rectangle(mask, (x, y), (x + width, y + height), color, -1)
        return cv2.addWeighted(image, 1, mask, 0.5, 0)
    elif shape == 'filled_square':
      # Draw a square using the bounding box coordinates
      x, y, width, height = cv2.boundingRect(np.array(mask_data, dtype=np.int32))
      color = (229, 231, 232)
      image = cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
      return image
    else:
      # Original polygon fill logic
      segments = [
          np.array(segment, dtype=np.int32).reshape((-1, 1, 2)) if np.array(segment, dtype=np.int32).shape[0] == 1 else np.array(segment, dtype=np.int32).reshape((1, -1, 2)) for segment in mask_data
      ]
      cv2.fillPoly(mask, segments, color)
      return cv2.addWeighted(image, 1, mask, 0.5, 0)

def visualize_tiled_annotations_sahi_orig(
    masked_save_path, tiled_img_path, tiled_annotations, 
    invalid_annotations, too_small_annotations, 
    cutoff_annotations, save_img = False, 
    plot_img = False, remove_cutoff = False):
  # Process each image
  comparison_imgs = sorted(glob.glob(f"{tiled_img_path}/*.png"))
  for image_path in comparison_imgs:
      # Get annotations for this image
      print(image_path)
      label = image_path.split('/')[-1][:-4]
      annotations = tiled_annotations[label]
      annotations_invalid = invalid_annotations[label]
      annotations_too_small = too_small_annotations[label]
      annotations_cutoff = cutoff_annotations[label]

      # Load the image, mask out invalid_polygons
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if os.path.exists(masked_save_path) is False:
          os.makedirs(masked_save_path)

      # Draw invalid masks as squares (we always want to mask out invalid masks)
      if len(annotations_invalid) > 0:
        for invalid_ann in annotations_invalid:
          color = (255, 0, 0)  # Red color for invalid polygons
          # Call draw_mask with shape='square' to draw a bounding box square
          image = draw_mask(image, invalid_ann, color, shape = 'filled_square', roi = background_value)
      
      #  Draw cut-off masks as squares (we may not want to mask these out)
      if len(annotations_cutoff) > 0:
        for cutoff_ann in annotations_cutoff:
          color = (128, 0, 128)  # Red color for invalid polygons
          # Call draw_mask with shape='square' to draw a bounding box square
          if remove_cutoff is True:
            image = draw_mask(image, cutoff_ann, color, shape = 'filled_square', roi = background_value)
            # Save masked images
            cv2.imwrite(f"{masked_save_path}/{label}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
          else: 
            # Save masked images before showing the cut-off ones in red for visualization purposes
            cv2.imwrite(f"{masked_save_path}/{label}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image = draw_mask(image, cutoff_ann, color, shape = 'polygon')
    
      # If there are no annotations, continue
      print(label)
      if len(annotations) <= 0:
        print("No annotations found")
        print()
        continue

      # Draw each mask
      for ann in annotations:
        segmentation = [np.array(seg).reshape(-1, 2).flatten().tolist() for seg in ann['segmentation']]
        # print(segmentation)
        color = (0, 255, 0)
        # if ann['area'] >= 100 and ann["area"] <= 110:
        image = draw_mask(image, segmentation, color, shape = 'polygon')

      for small_ann in annotations_too_small:
        # print(small_ann)
        color = (0, 0, 255)  # Blue color for "too small" polygons
        # Call draw_mask with shape='square' to draw a bounding box square
        image = draw_mask(image, small_ann, color, shape = 'polygon')

      print(f"Valid Annotations: {len(annotations)} | Invalid Annotations: {len(annotations_invalid)} | Cut-off Annotations: {len(annotations_cutoff)} | Annotations < 15 pixels: {len(annotations_too_small)}")
      print()

      if plot_img is True:
        # Display the image
        plt.figure()
        plt.imshow(image)
        plt.title(image_path)
        plt.axis('off')
        plt.show()

      if save_img:
        save_path = f"{masked_save_path}/visualizations"
        if os.path.exists(save_path) is False:
          os.mkdir(save_path)
        cv2.imwrite(f"{save_path}/{label}.png", image)

def visualize_tiled_annotations_sahi(
    masked_save_path, tiled_img_path, tiled_annotations, 
    invalid_annotations, too_small_annotations, 
    cutoff_annotations, save_img = False, 
    plot_img = False, remove_cutoff = False):
    
    # Process each image
    comparison_imgs = sorted(glob.glob(f"{tiled_img_path}/*.png"))
    for image_path in comparison_imgs:
        # Get annotations for this image
        print(image_path)
        label = image_path.split('/')[-1][:-4]
        annotations = tiled_annotations[label]
        annotations_invalid = invalid_annotations[label]
        annotations_too_small = too_small_annotations[label]
        annotations_cutoff = cutoff_annotations[label]

        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.exists(masked_save_path) is False:
            os.makedirs(masked_save_path)

        # Create a copy for masked version
        masked_image = image.copy()
        needs_masking = False

        # Mask out invalid regions in masked_image
        if len(annotations_invalid) > 0:
            needs_masking = True
            for invalid_ann in annotations_invalid:
                masked_image = draw_mask(masked_image, invalid_ann, color=(255, 0, 0), 
                                      shape='filled_square', roi=background_value)
        
        # Mask out cutoff regions if requested
        if len(annotations_cutoff) > 0 and remove_cutoff:
            needs_masking = True
            for cutoff_ann in annotations_cutoff:
                masked_image = draw_mask(masked_image, cutoff_ann, color=(128, 0, 128),
                                      shape='filled_square', roi=background_value)

        # Save the masked image (or original if no masking needed)
        if save_img:
            save_image = masked_image if needs_masking else image
            cv2.imwrite(f"{masked_save_path}/{label}.png", 
                       cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))

        # Create visualization image (start with masked version)
        viz_image = masked_image.copy()
        
        # Draw valid annotations for visualization
        for ann in annotations:
            segmentation = [np.array(seg).reshape(-1, 2).flatten().tolist() 
                          for seg in ann['segmentation']]
            viz_image = draw_mask(viz_image, segmentation, color=(0, 255, 0), 
                                shape='polygon')

        # Draw small annotations for visualization
        for small_ann in annotations_too_small:
            viz_image = draw_mask(viz_image, small_ann, color=(0, 0, 255), 
                                shape='polygon')

        # Draw cutoff annotations (if not removed) for visualization
        if len(annotations_cutoff) > 0 and not remove_cutoff:
            for cutoff_ann in annotations_cutoff:
                viz_image = draw_mask(viz_image, cutoff_ann, color=(128, 0, 128), 
                                    shape='polygon')

        print(f"Valid Annotations: {len(annotations)} | Invalid Annotations: {len(annotations_invalid)} | "
              f"Cut-off Annotations: {len(annotations_cutoff)} | "
              f"Annotations < 15 pixels: {len(annotations_too_small)}")
        print()

        # Save visualization
        if save_img:
            save_viz_path = f"{masked_save_path}/visualizations"
            if os.path.exists(save_viz_path) is False:
                os.makedirs(save_viz_path)
            cv2.imwrite(f"{save_viz_path}/{label}.png", viz_image)

        # Display if requested
        if plot_img:
            plt.figure()
            plt.imshow(viz_image)
            plt.title(image_path)
            plt.axis('off')
            plt.show()
            
    # Check images in dictionary vs images in folder
    img_files = os.listdir(masked_save_path)
    img_files = [f[:-4] for f in img_files if f.endswith('.png')] # Remove .png extension

    dict_keys = list(tiled_annotations.keys())

    if len(img_files) != len(dict_keys):
        print(f"Number of images mismatch - Dictionary: {len(dict_keys)} | Folder: {len(img_files)}")
    
    # Find images in dictionary but not in folder 
    dict_not_folder = set(dict_keys) - set(img_files)
    if dict_not_folder:
        print("Images in dictionary but not in folder:")
    for img in dict_not_folder:
        print(f"- {img}")

    # Find images in folder but not in dictionary
    folder_not_dict = set(img_files) - set(dict_keys)  
    if folder_not_dict:
        print("Images in folder but not in dictionary:")
    for img in folder_not_dict:
        print(f"- {img}")

    assert len(img_files) == len(dict_keys), "Number of images in folder does not match dictionary keys"
    assert set(img_files) == set(dict_keys), "Image names in folder do not match dictionary keys"

def merge_multi_polys(coco_data):
    merged_polys, merged_multi_polys = [], []
    for m_annot in coco_data['annotations']:
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
                        merged_polys.append(largest_single_poly)
                        m_annot['segmentation'] = [[list(i) for i in list(zip(*largest_single_poly.exterior.coords.xy))]]
                    else:  # Single Polygon
                        merged_polys.append(largest_poly)
                        m_annot['segmentation'] = [[list(i) for i in list(zip(*largest_poly.exterior.coords.xy))]]
                else:
                    # If no polygon geometries found, remove annotation
                    coco_data['annotations'].remove(m_annot)
                    continue
                    
            elif isinstance(union_result, MultiPolygon):
                merged_multi_polys.append(union_result)
                # Get the largest polygon from the MultiPolygon
                largest_poly = max(union_result.geoms, key=lambda x: x.area)
                merged_polys.append(largest_poly)
                m_annot['segmentation'] = [[list(i) for i in list(zip(*largest_poly.exterior.coords.xy))]]
                
            elif isinstance(union_result, Polygon):
                merged_polys.append(union_result)
                m_annot['segmentation'] = [[list(i) for i in list(zip(*union_result.exterior.coords.xy))]]
                
            else:  # LineString, Point, or other geometry types
                coco_data['annotations'].remove(m_annot)
                continue
        else:
            coco_data['annotations'].remove(m_annot)
            
    print(f"Number of merged polygons: {len(merged_polys)}")
    return coco_data

def parse_args():
    parser = argparse.ArgumentParser(description='Process SAHI dataset with specified base path')
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Base path to the SAHI dataset (e.g., /path/to/sahi_datasets/sahi_1536_ignore_neg)'
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load dataset JSON
    DATASET_BASE_PATH = args.dataset_path
    CROP_SZ = int(DATASET_BASE_PATH.split("/")[-1].split("_")[1])
    print(f"Crop size: {CROP_SZ}")

    for split in ["train", "val", "test"]:
        DATASET_JSON_PATH = f"{DATASET_BASE_PATH}/annotations/instances_{split}2017.json"
        with open(DATASET_JSON_PATH, "r") as f:
            sliced_coco_data = json.load(f)
        check_crop_szs(sliced_coco_data, CROP_SZ)

        # Set path for masked images
        base_mask_dir = f"{DATASET_BASE_PATH}/{split}2017_masked"
        if os.path.exists(base_mask_dir) is False:
            os.makedirs(base_mask_dir)

        # Now correct the annotations
        sahi_coco_data_fixed = merge_multi_polys(sliced_coco_data)

        # Post-process annotations by masking invalid/cutoff masks
        tiled_annotations, invalid_annotations, small_annotations, cutoff_annotations = get_sahi_tiled_annots(
            sahi_coco_data_fixed,
            remove_cutoff = False
        )
        visualize_tiled_annotations_sahi(
            masked_save_path = base_mask_dir, 
            tiled_img_path = f"{DATASET_BASE_PATH}/{split}2017",
            tiled_annotations = tiled_annotations,
            invalid_annotations = invalid_annotations,
            too_small_annotations = small_annotations,
            cutoff_annotations = cutoff_annotations,
            save_img = True,
            plot_img = False,
            remove_cutoff = False
        )

        with open(f"{DATASET_BASE_PATH}/{split}2017_masked.pkl", "wb") as f:
            pickle.dump(tiled_annotations, f)

        with open(f"{DATASET_BASE_PATH}/{split}2017_masked.pkl", "rb") as f:
            tiled_annotations = pickle.load(f)
        
        # if split == "train" and os.path.exists(f"{DATASET_BASE_PATH}_backup") is False:
        #     # Make backup of dataset folder; rename with "_orig" folder name suffix
        #     print(f"Backing up dataset: {DATASET_BASE_PATH} to {DATASET_BASE_PATH}_backup...")
        #     shutil.copytree(DATASET_BASE_PATH, f"{DATASET_BASE_PATH}_backup")
        #     print("Backup complete")
        #     print()
        
        print("Loading post-processed annotations...")
        with open(f"{DATASET_BASE_PATH}/{split}2017_masked.pkl", "rb") as f:
            tiled_annotations = pickle.load(f)
        print("Post-processed annotation loaded")
        print()

        # Create archive directory, if it doesn't exist 
        archive_path = f"{DATASET_BASE_PATH}/archive"
        if os.path.exists(archive_path) is False:
            os.makedirs(archive_path)
            
        # Check that post-processed tiles and annotations retain the same number of images 
        with open(DATASET_JSON_PATH, "r") as f:
            orig_tiled_annots = json.load(f)
        assert len(orig_tiled_annots['images']) == len(tiled_annotations.keys()), f"Number of images inconsistent after postprocessing - Orig: {len(orig_tiled_annots)} | After: {len(tiled_annotations.keys())}"
        img_folder = f"{DATASET_BASE_PATH}/{split}2017"
        assert len(orig_tiled_annots['images']) == len(os.listdir(img_folder)), f"Number of images inconsistent b/w image folder and JSON - JSON: {len(orig_tiled_annots)} | Image Folder: {len(os.listdir(img_folder))}"

        # Generate new annotation file 
        print("Generating new annotation JSON...")
        postprocessed_annots = []
        for k, v in tiled_annotations.items():
            postprocessed_annots.extend(v)
        print(f"Original Annotations: {len(orig_tiled_annots['annotations'])} | Post-processed annotations: {len(postprocessed_annots)}")
        
        postprocessed_tile_annots = orig_tiled_annots.copy()
        postprocessed_tile_annots['annotations'] = postprocessed_annots
        print("Generated new annotation JSON.")
        print()

        # Move files around:
        # Move annotations, train2017, and val2017 to archive folder
        shutil.move(f"{DATASET_BASE_PATH}/{split}2017", f"{DATASET_BASE_PATH}/archive/{split}2017_raw")
        print(f"Moved {DATASET_BASE_PATH}/{split}2017 to {DATASET_BASE_PATH}/archive/{split}2017_raw")

        shutil.move(f"{DATASET_BASE_PATH}/annotations/instances_{split}2017.json", f"{DATASET_BASE_PATH}/archive/instances_{split}2017_raw.json")
        print(f"Moved {DATASET_BASE_PATH}/annotations/instances_{split}2017.json to {DATASET_BASE_PATH}/archive/instances_{split}2017_raw.json")

        # Rename train2017_masked and val2017_masked to train2017 and val2017
        os.rename(f"{DATASET_BASE_PATH}/{split}2017_masked", f"{DATASET_BASE_PATH}/{split}2017")
        print(f"Renamed {DATASET_BASE_PATH}/{split}2017_masked to {DATASET_BASE_PATH}/{split}2017")

        # Save postprocessed JSON file
        with open(DATASET_JSON_PATH, "w") as f:
            json.dump(postprocessed_tile_annots, f, indent=4)
        print(f"Saved postprocessed JSON file to {DATASET_JSON_PATH}")
        
        # Move visualization folders to archive 
        shutil.move(f"{DATASET_BASE_PATH}/{split}2017/visualizations", f"{DATASET_BASE_PATH}/archive/masked_visualizations_{split}2017")
        print(f"Moved {DATASET_BASE_PATH}/{split}2017/visualizations to {DATASET_BASE_PATH}/archive/masked_visualizations_{split}2017")

        # Move .PKL to archive 
        shutil.move(f"{DATASET_BASE_PATH}/{split}2017_masked.pkl", f"{DATASET_BASE_PATH}/archive/{split}2017_masked.pkl")
        print(f"Moved {DATASET_BASE_PATH}/{split}2017_masked.pkl to {DATASET_BASE_PATH}/archive/{split}2017_masked.pkl")

        # Remove archive folder 
        shutil.rmtree(f"{DATASET_BASE_PATH}/archive")
        print(f"Removed {DATASET_BASE_PATH}/archive")

        # # Post file operation checks
        # assert len(os.listdir)

if __name__ == "__main__":
    main()