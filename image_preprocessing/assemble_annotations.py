import json
import os
import cv2
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import copy
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import argparse

def load_coco_json(file_path):
    """Loads a COCO JSON file."""
    print(f"Loading COCO JSON from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: COCO JSON file not found at {file_path}")
        return None
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_coco_json(data, file_path):
    """Saves data to a COCO JSON file."""
    print(f"Saving COCO JSON to: {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def filter_annotations_by_category(coco_data, target_category_name):
    """Filters annotations to keep only those matching target_category_name."""
    if 'annotations' not in coco_data or 'name' not in coco_data['annotations'][0]:
        print("Warning: 'annotations' key not found or annotations do not have 'name' key. Skipping category filtering.")
        return coco_data
    
    original_count = len(coco_data['annotations'])
    coco_data['annotations'] = [
        annot for annot in coco_data['annotations'] if annot.get('name') == target_category_name
    ]
    print(f"Filtered annotations by category '{target_category_name}'. Kept {len(coco_data['annotations'])} out of {original_count}.")
    return coco_data

def update_image_paths_and_dimensions(coco_data, content_root_dir):
    """Updates image paths, dimensions, and filenames in COCO data."""
    print(f"Updating image paths and dimensions using content root: {content_root_dir}")
    images_to_remove_indices = []
    for idx, img_entry in enumerate(coco_data['images']):
        updated = False
        # Try original path
        try:
            # Assuming 'toras_path' is the key containing the relative path from content_root_dir
            if 'toras_path' not in img_entry:
                print(f"Warning: 'toras_path' not found for image entry: {img_entry.get('id', 'Unknown ID')}. Skipping.")
                images_to_remove_indices.append(idx)
                continue

            img_entry['path'] = img_entry['toras_path'] # Store the relative path
            full_image_path = os.path.join(content_root_dir, img_entry['path'])
            
            # This is required as all the toras_paths are for 'batch-1', 
            # yet in the GDrive folder and in TORAS we have 'batch-1' and 'batch-2'
            if not os.path.exists(full_image_path):
                # Try replacing 'batch-1' with 'batch-2' as per original logic
                alt_toras_path = img_entry['toras_path'].replace('batch-1', 'batch-2')
                alt_full_image_path = os.path.join(content_root_dir, alt_toras_path)
                if os.path.exists(alt_full_image_path):
                    print(f"Found image at alternate path: {alt_full_image_path} (was {full_image_path})")
                    img_entry['toras_path'] = alt_toras_path # Update toras_path if using alternative
                    img_entry['path'] = alt_toras_path
                    full_image_path = alt_full_image_path
                else:
                    print(f"Image not found: {full_image_path} (and alternate also not found: {alt_full_image_path})")
                    images_to_remove_indices.append(idx)
                    continue
            
            image_data = cv2.imread(full_image_path)
            if image_data is None:
                print(f"Could not read image: {full_image_path}")
                images_to_remove_indices.append(idx)
                continue

            img_entry['height'] = image_data.shape[0]
            img_entry['width'] = image_data.shape[1]
            img_entry['file_name'] = os.path.basename(img_entry['toras_path']) # Use basename of the path
            updated = True

        except Exception as e:
            print(f"Error processing image {img_entry.get('toras_path', 'Unknown path')}: {e}")
            images_to_remove_indices.append(idx)
        
        if not updated:
             print(f"Failed to update image entry: {img_entry.get('id', 'Unknown ID')}, path: {img_entry.get('toras_path', 'N/A')}")


    # Remove images that couldn't be processed, iterating in reverse to maintain indices
    for index in sorted(images_to_remove_indices, reverse=True):
        print(f"Removing image entry at index {index} due to processing errors: {coco_data['images'][index].get('toras_path', 'N/A')}")
        del coco_data['images'][index]
    
    # Filter annotations to only include those for remaining images
    remaining_image_ids = {img['id'] for img in coco_data['images']}
    if 'annotations' in coco_data:
        original_annot_count = len(coco_data['annotations'])
        coco_data['annotations'] = [
            ann for ann in coco_data['annotations'] if ann.get('image_id') in remaining_image_ids
        ]
        print(f"Filtered annotations based on remaining images. Kept {len(coco_data['annotations'])} from {original_annot_count}.")

    print(f"Number of unique bulk images after path update: {len(set(im['file_name'][:-9] for im in coco_data['images'] if len(im['file_name']) > 9))}")
    return coco_data

def merge_multi_polys(coco_data):
    print("Merging multi-polygons...")
    merged_polys_count = 0
    annotations_to_keep = []
    
    for m_annot in coco_data.get('annotations', []):
        polygons_shapely = []
        if not m_annot.get('segmentation'): # Skip if no segmentation
            continue

        for polygon_coords in m_annot['segmentation']:
            segment_coords_np = np.array(polygon_coords).reshape(-1, 2)
            if len(segment_coords_np) >= 3: # A polygon needs at least 3 points
                try:
                    polygons_shapely.append(make_valid(Polygon(segment_coords_np)))
                except Exception as e:
                    print(f"Warning: Could not create/validate polygon for annotation {m_annot.get('id')}: {e}")
                    continue # Skip this segment
            # else: # Not enough points for a polygon
            #     print(f"Warning: Annotation {m_annot.get('id')} has a segment with < 3 points. Skipping segment.")

        if not polygons_shapely: # No valid polygons found for this annotation
            # print(f"Warning: Annotation {m_annot.get('id')} has no valid polygons after processing. Removing annotation.")
            continue 

        try:
            union_result = unary_union(polygons_shapely)
        except Exception as e:
            print(f"Warning: unary_union failed for annotation {m_annot.get('id')}: {e}. Skipping annotation.")
            continue
        
        final_polygon_for_annot = None
        if isinstance(union_result, GeometryCollection):
            poly_geoms = [geom for geom in union_result.geoms if isinstance(geom, (Polygon, MultiPolygon))]
            if poly_geoms:
                largest_poly_geom = max(poly_geoms, key=lambda x: x.area)
                if isinstance(largest_poly_geom, MultiPolygon):
                    final_polygon_for_annot = max(largest_poly_geom.geoms, key=lambda x: x.area)
                else:
                    final_polygon_for_annot = largest_poly_geom
        elif isinstance(union_result, MultiPolygon):
            final_polygon_for_annot = max(union_result.geoms, key=lambda x: x.area)
        elif isinstance(union_result, Polygon):
            final_polygon_for_annot = union_result
        
        if final_polygon_for_annot and final_polygon_for_annot.area > 0:
            # Ensure exterior coords are correctly formatted for COCO
            exterior_coords = list(zip(*final_polygon_for_annot.exterior.coords.xy))
            # COCO expects a list of lists of coordinates, e.g. [[x1,y1,x2,y2,...]] or [[[x1,y1],[x2,y2],...]]
            # The original script uses [[list(i) for i in list(zip(*largest_single_poly.exterior.coords.xy))]]
            # which results in [[[x1,y1],[x2,y2],...]]
            # Let's reformat to simple list [x1,y1,x2,y2,...] for SAHI/standard COCO.
            # If [[[x1,y1],...]] is truly needed, this needs adjustment.
            # Based on `segment_coords = np.array(annot['segmentation'][polygon_idx])` later, 
            # it seems the original was expecting a flat list per polygon.
            # Let's stick to a single polygon per annotation, with flat coordinates.
            m_annot['segmentation'] = [np.array(exterior_coords).ravel().tolist()]
            merged_polys_count +=1
            annotations_to_keep.append(m_annot)
        # else:
            # print(f"Warning: Annotation {m_annot.get('id')} resulted in empty or invalid geometry after union. Removing annotation.")
    
    coco_data['annotations'] = annotations_to_keep
    print(f"Number of annotations with successfully merged/processed polygons: {merged_polys_count}")
    return coco_data

def shift_annotations(coco_data_annots, metadata_tile_info, img_id_for_annot, min_area_size):
    shifted_annotations_list = []
    invalid_polygons_list = []
    invalid_poly_count = 0
    total_segments_processed = 0

    # metadata_tile_info should be (metadata_y_shift, metadata_x_shift) for the current tile
    metadata_y_shift, metadata_x_shift = metadata_tile_info

    # Annotations are already filtered for the current image_id implicitly by project_tiled_annots
    for annot_orig in coco_data_annots:
        annot = copy.deepcopy(annot_orig) # Work on a copy
        revised_coords_for_annot = []
        shapely_polygons_for_annot = []
        
        if not annot.get('segmentation'):
            continue
        total_segments_processed += len(annot['segmentation'])

        for segment_list_coords in annot['segmentation']: # segment_list_coords is [x1,y1,x2,y2,...]
            segment_coords_np = np.array(segment_list_coords).reshape(-1, 2)
            segment_coords_np[:, 0] += metadata_x_shift
            segment_coords_np[:, 1] += metadata_y_shift
            
            if segment_coords_np.shape[0] >= 3:
                poly = Polygon(segment_coords_np)
                if poly.is_valid and poly.area > 0: # Added area check
                    # Store as flat list of numbers for COCO segmentation
                    revised_coords_for_annot.append(segment_coords_np.ravel().tolist())
                    shapely_polygons_for_annot.append(poly)
                else:
                    invalid_poly_count += 1
                    invalid_polygons_list.append(segment_coords_np.tolist()) 
            else: # Not enough points
                invalid_poly_count +=1 
                invalid_polygons_list.append(segment_coords_np.tolist())


        if shapely_polygons_for_annot: # If any valid polygons were formed for this annotation
            # If multiple polygons resulted from one original annotation (e.g. from a MultiPolygon that got split)
            # We need to decide how to handle them. Original script seems to make one MultiPolygon.
            # Let's assume the merge_multi_polys function already ensured one polygon per annotation.
            # So, shapely_polygons_for_annot should ideally have one item.
            
            # For safety, let's union them if there are multiple valid pieces from one original annotation
            # This can happen if the original annotation was already a multipolygon that merge_multi_polys didn't fully consolidate
            # or if shifting created disjoint valid parts.
            if len(shapely_polygons_for_annot) > 1:
                 current_annot_geometry = unary_union(shapely_polygons_for_annot)
            else:
                 current_annot_geometry = shapely_polygons_for_annot[0]

            if isinstance(current_annot_geometry, MultiPolygon):
                # If it's a multipolygon, take the largest part as per original script's theme
                current_annot_geometry = max(current_annot_geometry.geoms, key=lambda p: p.area)

            if isinstance(current_annot_geometry, Polygon) and current_annot_geometry.area >= min_area_size:
                x, y, max_x, max_y = current_annot_geometry.bounds
                width = max_x - x
                height = max_y - y
                
                annot['area'] = current_annot_geometry.area
                annot['bbox'] = [x, y, width, height]
                # Segmentation should be the exterior of this final polygon
                annot['segmentation'] = [np.array(list(current_annot_geometry.exterior.coords)).ravel().tolist()]
                annot['image_id'] = img_id_for_annot # Assign the new bulk image ID
                shifted_annotations_list.append(annot)
            # else: # Area too small or not a polygon
                # print(f"Annotation part discarded (area < {min_area_size} or not polygon) for original annot ID {annot_orig.get('id')}")

    # This print was outside the loop in original, might be better per image
    # print(f"Invalid polygons during shift: {invalid_poly_count} | Total segments processed: {total_segments_processed}")
    return shifted_annotations_list, invalid_polygons_list, invalid_poly_count, total_segments_processed

def draw_mask(image, mask_data_segment, color, shape='polygon'):
    # mask_data_segment should be a single list of coords [x1,y1,x2,y2,...] for one polygon
    mask = np.zeros_like(image)
    points = np.array(mask_data_segment).reshape(-1, 2).astype(np.int32)

    if shape == 'square':
        if len(points) > 0: # Need points to calculate bounding box
            x, y, w, h = cv2.boundingRect(points)
            cv2.rectangle(mask, (x, y), (x + w, y + h), color, -1)
        else: # No points, can't draw a square
            return image # Return original image
    else: # Polygon
        if len(points) >=3: # Need at least 3 points for a polygon
             cv2.fillPoly(mask, [points], color)
        # else: # Not enough points, can't draw polygon
            # return image # Return original image

    return cv2.addWeighted(image, 1, mask, 0.5, 0)


def plot_shifted_annotations_on_image(shifted_annotations, invalid_polygons_coords, 
                                      bulk_image_path, img_display_name, 
                                      do_save_img, do_plot_annotations, output_save_path):
    if not (do_plot_annotations or do_save_img):
        return

    if not os.path.exists(bulk_image_path):
        print(f"Error: Bulk image not found for plotting: {bulk_image_path}")
        return

    image = cv2.imread(bulk_image_path)
    if image is None:
        print(f"Error: Could not read bulk image for plotting: {bulk_image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    category_colors = {
        "b": (0, 255, 0),    # GREEN = BUG
        "d": (255, 0, 143),  # PURPLE = DEBRIS
        "u": (0, 165, 255),  # ORANGE = UNKNOWN or EDGE
        "e": (0, 0, 255),    # RED = ENTITY
        "default": (128, 128, 128) # GRAY for unknown categories
    }

    # Draw each valid shifted mask
    for ann in shifted_annotations:
        color = category_colors.get(ann.get('name', 'default'), category_colors["default"])
        for seg_coords in ann['segmentation']: # ann['segmentation'] is list of lists of coords
            image = draw_mask(image, seg_coords, color, shape='polygon')

    # Draw invalid polygons as red squares (using their original coordinates before shifting for bbox)
    # The invalid_polygons_coords are already shifted.
    for invalid_poly_coord_list in invalid_polygons_coords: # list of coordinate lists
        image = draw_mask(image, invalid_poly_coord_list, (255, 0, 0), shape='square')
        
    if do_plot_annotations:
        plt.imshow(image)
        plt.title(img_display_name)
        plt.axis('off')
        plt.show()

    if do_save_img:
        if not output_save_path:
            print("Error: Save path not provided for saving image.")
            return
        os.makedirs(output_save_path, exist_ok=True)
        save_image_path = os.path.join(output_save_path, f"{img_display_name}.png")
        cv2.imwrite(save_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated image to: {save_image_path}")


def project_tiled_annotations_to_bulk(metadata_path, coco_data_all_annots, bulk_img_name,
                                      bulk_image_id_hash,
                                      min_shifted_area,
                                      content_root_dir, bulk_images_dir_name,
                                      do_save_img, do_plot_annotations, output_save_dir_for_imgs):
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}. Skipping projection for {bulk_img_name}.")
        return [], [], 0, 0
    
    metadata = np.load(metadata_path) # (y_min, y_max, x_min, x_max) for each tile

    # Find all tile image entries in coco_data['images'] that belong to this bulk_img_name
    # Tile images are identified if bulk_img_name is part of their 'path'
    tile_img_entries = [
        img_entry for img_entry in coco_data_all_annots['images'] 
        if bulk_img_name in img_entry.get('path', '')
    ]

    if not tile_img_entries:
        print(f"No annotated image tiles found for bulk image: {bulk_img_name}")
        return [], [], 0, 0

    all_shifted_annotations_for_bulk = []
    all_invalid_polygons_for_bulk = []
    total_invalid_count_for_bulk = 0
    total_segments_processed_for_bulk = 0

    for tile_img_entry in tile_img_entries:
        tile_id = tile_img_entry['id']
        tile_file_name = tile_img_entry['file_name']
        
        try:
            # Tile index is the number at the end of the filename, e.g., "image_tile_0.jpg" -> 0
            tile_idx = int(os.path.splitext(tile_file_name)[0].split('_')[-1])
        except (IndexError, ValueError) as e:
            print(f"Could not parse tile index from filename: {tile_file_name} for bulk {bulk_img_name}. Error: {e}. Skipping tile.")
            continue
        
        if tile_idx >= len(metadata):
            print(f"Tile index {tile_idx} out of bounds for metadata (len: {len(metadata)}) for {tile_file_name}. Skipping tile.")
            continue

        # Get annotations corresponding to this specific tile_id
        annotations_for_this_tile = [
            ann for ann in coco_data_all_annots['annotations'] if ann.get('image_id') == tile_id
        ]
        
        if not annotations_for_this_tile:
            # print(f"No annotations found for tile: {tile_file_name} (ID: {tile_id})")
            continue

        # Metadata for this tile: (y_min, y_max, x_min, x_max)
        # We need the top-left corner for shifting: (y_min, x_min)
        metadata_y_shift = metadata[tile_idx][0]
        metadata_x_shift = metadata[tile_idx][2]
        
        shifted_annots, invalid_polys, inv_count, seg_count = shift_annotations(
            annotations_for_this_tile, 
            (metadata_y_shift, metadata_x_shift), 
            bulk_image_id_hash, # The new image_id for the bulk image
            min_shifted_area
        )
        all_shifted_annotations_for_bulk.extend(shifted_annots)
        all_invalid_polygons_for_bulk.extend(invalid_polys) # These are coordinate lists
        total_invalid_count_for_bulk += inv_count
        total_segments_processed_for_bulk += seg_count

    # Plotting/saving uses the consolidated list of annotations for the bulk image
    if do_plot_annotations or do_save_img:
        bulk_image_full_path = os.path.join(content_root_dir, bulk_images_dir_name, bulk_img_name, f"{bulk_img_name}.jpg")
        plot_shifted_annotations_on_image(
            all_shifted_annotations_for_bulk, 
            all_invalid_polygons_for_bulk, # Pass list of coordinate lists
            bulk_image_full_path,
            bulk_img_name, 
            do_save_img, 
            do_plot_annotations, 
            output_save_dir_for_imgs
        )
    
    return all_shifted_annotations_for_bulk, all_invalid_polygons_for_bulk, total_invalid_count_for_bulk, total_segments_processed_for_bulk


def get_image_names_from_dir(directory_path):
    """Gets base names of subdirectories (assumed to be image names)."""
    if not os.path.isdir(directory_path):
        print(f"Warning: Directory not found for get_image_names_from_dir: {directory_path}")
        return []
    # Assuming each direct subdirectory of 'directory_path' is an image folder
    img_folders = glob.glob(os.path.join(directory_path, "*"))
    img_names = [os.path.basename(folder) for folder in img_folders if os.path.isdir(folder)]
    return img_names


def create_final_coco_structure(output_dir):
    """Creates the base directory structure for COCO."""
    print(f"Creating final COCO directory structure at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for folder in ["annotations", "train2017", "val2017", "test2017"]:
        folder_path = os.path.join(output_dir, folder)
        os.makedirs(folder_path, exist_ok=True)


def split_coco_dataset_by_prefix(assembled_coco_path, final_coco_output_dir, val_prefixes, test_prefixes):
    """Splits the assembled COCO dataset based on filename prefixes."""
    print(f"Splitting COCO dataset from {assembled_coco_path} into {final_coco_output_dir}")
    coco_data = load_coco_json(assembled_coco_path)
    if coco_data is None: return

    images = coco_data['images']
    all_annotations = coco_data['annotations'] # Renamed to avoid conflict

    image_groups = {} # Group images by their prefix (filename without extension)
    for image in images:
        prefix = os.path.splitext(image['file_name'])[0]
        if prefix not in image_groups:
            image_groups[prefix] = []
        image_groups[prefix].append(image)

    all_file_prefixes = set(image_groups.keys())
    
    # Ensure val and test prefixes are sets for efficient lookup
    val_prefixes_set = set(val_prefixes)
    test_prefixes_set = set(test_prefixes)

    # Determine train prefixes
    train_prefixes_set = all_file_prefixes - val_prefixes_set - test_prefixes_set
    
    # Verify no overlap between val and test if they were user-provided
    if val_prefixes_set.intersection(test_prefixes_set):
        print(f"Warning: Overlap found between validation and test prefixes: {val_prefixes_set.intersection(test_prefixes_set)}")
    # Verify all provided val/test prefixes exist in the dataset
    for p_set, name in [(val_prefixes_set, "validation"), (test_prefixes_set, "test")]:
        missing = p_set - all_file_prefixes
        if missing:
            print(f"Warning: The following {name} prefixes are not found in the dataset: {missing}")
        p_set.intersection_update(all_file_prefixes) # Keep only existing prefixes


    print(f"Total unique image prefixes: {len(all_file_prefixes)}")
    print(f"Train prefixes count: {len(train_prefixes_set)}")
    print(f"Validation prefixes count: {len(val_prefixes_set)}")
    print(f"Test prefixes count: {len(test_prefixes_set)}")

    def collect_data_for_split(selected_prefixes):
        split_images = []
        split_annotations = []
        image_ids_in_split = set()

        for prefix_key in selected_prefixes:
            if prefix_key in image_groups:
                img_list_for_prefix = image_groups[prefix_key]
                split_images.extend(img_list_for_prefix)
                for img in img_list_for_prefix:
                    image_ids_in_split.add(img['id'])
            else:
                print(f"Warning: Prefix {prefix_key} specified for split not found in image_groups.")
        
        # Collect annotations for the images in this split
        for annotation in all_annotations:
            if annotation['image_id'] in image_ids_in_split:
                split_annotations.append(annotation)
        return split_images, split_annotations

    train_images, train_annotations = collect_data_for_split(train_prefixes_set)
    valid_images, valid_annotations = collect_data_for_split(val_prefixes_set)
    test_images, test_annotations = collect_data_for_split(test_prefixes_set)

    def create_coco_split_file(images_list, annotations_list, output_filename_base):
        new_coco_dataset = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "images": images_list,
            "annotations": annotations_list,
            "categories": coco_data['categories'] # Crucial: categories must be present
        }
        output_path = os.path.join(final_coco_output_dir, "annotations", f"instances_{output_filename_base}2017.json")
        save_coco_json(new_coco_dataset, output_path)
        return new_coco_dataset # Return the data for further processing

    train_data_split = create_coco_split_file(train_images, train_annotations, "train")
    val_data_split = create_coco_split_file(valid_images, valid_annotations, "val")
    test_data_split = create_coco_split_file(test_images, test_annotations, "test")
    
    print("Dataset splitting complete.")
    return train_data_split, val_data_split, test_data_split


def remove_empty_annotations_from_dataset(data):
    """Removes annotations that have no 'segmentation' or empty 'segmentation' list."""
    if 'annotations' not in data:
        return data
    
    original_count = len(data['annotations'])
    valid_annotations = []
    removed_ctr = 0
    for annotation in data['annotations']:
        if 'segmentation' in annotation and annotation['segmentation']: # Checks for key and non-empty list
            valid_annotations.append(annotation)
        else:
            removed_ctr += 1
    data['annotations'] = valid_annotations
    if removed_ctr > 0:
        print(f"Number of empty/invalid segmentations removed: {removed_ctr} (out of {original_count})")
    return data

def generate_agnostic_labels_for_dataset(coco_data_orig, agnostic_category_name="b", agnostic_category_id=1, supercategory="insect"):
    """Converts dataset annotations to a single agnostic category."""
    agnostic_coco_data = copy.deepcopy(coco_data_orig)
    agnostic_coco_data['categories'] = [{
        "id": agnostic_category_id,
        "name": agnostic_category_name,
        "supercategory": supercategory
    }]
    for annot in agnostic_coco_data.get('annotations', []):
        annot['category_id'] = agnostic_category_id
    return agnostic_coco_data


def copy_images_to_final_coco_split_folder(coco_split_data, split_name, content_root_dir, final_coco_output_dir):
    """Copies images listed in a COCO split to the corresponding final folder."""
    print(f"Copying images for {split_name} split...")
    destination_dir = os.path.join(final_coco_output_dir, f"{split_name}2017")
    os.makedirs(destination_dir, exist_ok=True)
    
    copied_count = 0
    failed_count = 0
    for img_entry in coco_split_data.get('images', []):
        # 'path' in img_entry should be the relative path from content_root_dir to the *bulk* image
        # e.g., "bulk_batch_1_and_2/IMAGE_NAME_X/IMAGE_NAME_X.jpg"
        # 'file_name' should be the simple file name, e.g., "IMAGE_NAME_X.jpg"
        if 'path' not in img_entry or 'file_name' not in img_entry:
            print(f"Warning: Image entry missing 'path' or 'file_name': {img_entry.get('id')}. Skipping copy.")
            failed_count +=1
            continue

        source_image_path = os.path.join(content_root_dir, img_entry['path'])
        destination_image_path = os.path.join(destination_dir, img_entry['file_name'])
        
        if os.path.exists(source_image_path):
            try:
                shutil.copyfile(source_image_path, destination_image_path)
                copied_count +=1
            except Exception as e:
                print(f"Error copying {source_image_path} to {destination_image_path}: {e}")
                failed_count +=1
        else:
            print(f"Source image not found, cannot copy: {source_image_path}")
            failed_count +=1
    print(f"Finished copying for {split_name}: {copied_count} copied, {failed_count} failed/skipped.")


def main(args):
    # 1. Load initial COCO JSON
    coco_format_data = load_coco_json(args.input_coco_json)
    if coco_format_data is None:
        return

    # 2. Filter annotations by target category name (e.g., 'b')
    if args.target_category_name:
        coco_format_data = filter_annotations_by_category(coco_format_data, args.target_category_name)

    # 3. Update image paths and dimensions
    # This step reads tile images, so content_root should point to where tile images are (e.g. batch-1/IMG/IMG_tile_0.jpg)
    coco_format_data = update_image_paths_and_dimensions(coco_format_data, args.content_root_dir)
    if not coco_format_data.get('images'):
        print("No images found after path updates. Exiting.")
        return

    # 4. Merge multi-polygons within annotations
    coco_format_data = merge_multi_polys(coco_format_data)
    if not coco_format_data.get('annotations'):
        print("No annotations remaining after merging multi-polygons. Exiting.")
        return


    # 5. Project tiled annotations to bulk images
    exclusion_list = set(args.exclusion_list.split(',') if args.exclusion_list else [])
    
    all_projected_annotations = []
    # Consolidated list of invalid polygons for summary, if needed
    # all_invalid_projected_polygons_info = [] 

    batch_dirs_to_process = []
    if args.batch1_data_dir_name:
        batch_dirs_to_process.append(args.batch1_data_dir_name)
    if args.batch2_data_dir_name:
        batch_dirs_to_process.append(args.batch2_data_dir_name)

    total_invalid_polygons_overall = 0
    total_segments_processed_overall = 0

    for batch_dir_name in batch_dirs_to_process:
        batch_full_path = os.path.join(args.content_root_dir, batch_dir_name)
        print(f"\nProcessing batch: {batch_dir_name} from {batch_full_path}")
        # Image names are assumed to be subfolder names within the batch directory
        bulk_image_names_in_batch = get_image_names_from_dir(batch_full_path)
        
        for bulk_img_name in bulk_image_names_in_batch:
            if bulk_img_name in exclusion_list:
                print(f"Skipping excluded image: {bulk_img_name}")
                continue
            
            print(f"  Projecting annotations for bulk image: {bulk_img_name}")
            metadata_file_path = os.path.join(batch_full_path, bulk_img_name, f"{bulk_img_name}_tile_metadata.npy")
            
            # The image_id for the new *bulk* image annotation will be a hash of its name
            bulk_image_id_hash = hash(bulk_img_name)

            shifted_annots, _, invalid_count, seg_count = project_tiled_annotations_to_bulk(
                metadata_path=metadata_file_path,
                coco_data_all_annots=coco_format_data, # Pass the main COCO data (contains tile annots)
                bulk_img_name=bulk_img_name,
                bulk_image_id_hash=bulk_image_id_hash,
                min_shifted_area=args.min_shifted_area,
                content_root_dir=args.content_root_dir, # For finding the bulk .jpg image itself
                bulk_images_dir_name=args.bulk_images_dir_name, # Subdir under content_root for bulk .jpgs
                do_save_img=args.save_intermediate_images,
                do_plot_annotations=args.plot_intermediate_annotations,
                output_save_dir_for_imgs=args.output_dir_processed_data # Where to save plotted .pngs
            )
            all_projected_annotations.extend(shifted_annots)
            total_invalid_polygons_overall += invalid_count
            total_segments_processed_overall += seg_count
    
    print(f"\nOverall invalid polygons during projection: {total_invalid_polygons_overall} out of {total_segments_processed_overall} segments processed.")
    print(f"Total projected annotations gathered: {len(all_projected_annotations)}")

    for ann in all_projected_annotations:
        ann['iscrowd'] = 0 # Set iscrowd for all new annotations

    # 6. Assemble the new COCO dataset for bulk images
    # Create the 'images' list for the assembled COCO data
    processed_bulk_images_coco_list = []
    # Get unique bulk image IDs from the projected annotations
    # These are hashes of bulk image names
    unique_bulk_image_ids_from_annots = sorted(list(set(ann['image_id'] for ann in all_projected_annotations)))
    
    # Need to map these IDs back to names to construct paths and read dimensions
    # This requires knowing which bulk_img_name corresponds to which hash.
    # A more robust way would be to collect bulk_img_name along with its hash earlier.
    # For now, assuming we can iterate through all possible bulk images again to build this.
    
    # Re-iterate bulk images to create the 'images' field for the new COCO JSON
    all_possible_bulk_images = []
    for batch_dir_name in batch_dirs_to_process:
         batch_full_path = os.path.join(args.content_root_dir, batch_dir_name)
         all_possible_bulk_images.extend(get_image_names_from_dir(batch_full_path))
    
    # Filter by those that actually have annotations and are not excluded
    relevant_bulk_image_names = set()
    for ann in all_projected_annotations:
        # This relies on a way to get back from ann['image_id'] (hash) to the name.
        # Let's find the name whose hash matches ann['image_id']
        for name_candidate in all_possible_bulk_images:
            if hash(name_candidate) == ann['image_id']:
                relevant_bulk_image_names.add(name_candidate)
                break
    
    print(f"Relevant bulk image names with annotations: {len(relevant_bulk_image_names)}")
    
    annotated_bulk_image_paths_for_coco = [] # Store paths for later check

    for bulk_img_name in sorted(list(relevant_bulk_image_names)):
        if bulk_img_name in exclusion_list: continue

        img_dict = {}
        img_dict['id'] = hash(bulk_img_name)
        img_dict['file_name'] = f"{bulk_img_name}.jpg" # Final filename in COCO structure

        # Path to the original bulk .jpg image to get its dimensions
        # This path is relative to content_root_dir
        original_bulk_jpg_rel_path = os.path.join(args.bulk_images_dir_name, bulk_img_name, f"{bulk_img_name}.jpg")
        original_bulk_jpg_full_path = os.path.join(args.content_root_dir, original_bulk_jpg_rel_path)
        
        if os.path.exists(original_bulk_jpg_full_path):
            loaded_img_data = cv2.imread(original_bulk_jpg_full_path)
            if loaded_img_data is not None:
                img_dict['height'] = loaded_img_data.shape[0]
                img_dict['width'] = loaded_img_data.shape[1]
                img_dict['path'] = original_bulk_jpg_rel_path # Store the relative path for copying later
                processed_bulk_images_coco_list.append(img_dict)
                annotated_bulk_image_paths_for_coco.append(original_bulk_jpg_full_path)
            else:
                print(f"Warning: Could not read bulk image {original_bulk_jpg_full_path} to get dimensions.")
        else:
            print(f"Warning: Original bulk image not found at {original_bulk_jpg_full_path} for COCO 'images' entry.")
            
    print(f"Number of bulk images added to new COCO 'images' list: {len(processed_bulk_images_coco_list)}")

    # Basic assertion from original script (can be made more flexible)
    if args.expected_annotated_images > 0:
        if len(processed_bulk_images_coco_list) != args.expected_annotated_images:
             print(f"Warning: Number of annotated bulk images ({len(processed_bulk_images_coco_list)}) "
                   f"does not match expected ({args.expected_annotated_images}). Check exclusion list or processing.")
        else:
            print(f"Number of annotated bulk images ({len(processed_bulk_images_coco_list)}) matches expected count.")


    # Verify all annotation image_ids are present in the new 'images' list
    final_image_ids_in_coco = {img['id'] for img in processed_bulk_images_coco_list}
    annotations_image_ids = {ann['image_id'] for ann in all_projected_annotations}
    if not final_image_ids_in_coco == annotations_image_ids:
        print("Error: Mismatch between image IDs in final 'images' list and 'annotations' list.")
        print(f"  IDs in 'images' but not 'annotations': {final_image_ids_in_coco - annotations_image_ids}")
        print(f"  IDs in 'annotations' but not 'images': {annotations_image_ids - final_image_ids_in_coco}")
        # Potentially filter annotations to only include those with images
        all_projected_annotations = [ann for ann in all_projected_annotations if ann['image_id'] in final_image_ids_in_coco]
        print(f"  Filtered annotations to {len(all_projected_annotations)} to match available images.")


    assembled_coco_data = {
        'info': coco_format_data.get('info', {}), # From original input
        'licenses': coco_format_data.get('licenses', []), # From original input
        'categories': coco_format_data.get('categories', []), # From original input (filtered by target_category if applicable)
        'images': processed_bulk_images_coco_list,
        'annotations': all_projected_annotations
    }
    
    assembled_coco_filename = f"coco_format_assembled_{args.target_category_name}.json"
    assembled_coco_full_path = os.path.join(args.output_dir_processed_data, assembled_coco_filename)
    save_coco_json(assembled_coco_data, assembled_coco_full_path)


    # 7. Create final COCO directory structure (train/val/test)
    create_final_coco_structure(args.final_output_coco_dir)

    # 8. Split the assembled dataset
    val_prefixes = args.val_prefixes_list.split(',') if args.val_prefixes_list else []
    test_prefixes = args.test_prefixes_list.split(',') if args.test_prefixes_list else []
    
    train_data, val_data, test_data = split_coco_dataset_by_prefix(
        assembled_coco_full_path, 
        args.final_output_coco_dir, 
        val_prefixes, 
        test_prefixes
    )

    # 9. Post-split processing (remove empty annotations, generate agnostic labels)
    datasets_to_process = {
        "train": train_data, 
        "val": val_data, 
        "test": test_data
    }
    
    processed_datasets_taxonomic = {}
    processed_datasets_agnostic = {}

    for split_name, data in datasets_to_process.items():
        print(f"\nPost-processing {split_name} data...")
        data = remove_empty_annotations_from_dataset(data) # Modifies in place if dict is passed
        
        # Save taxonomic version
        tax_path = os.path.join(args.final_output_coco_dir, "annotations", f"instances_{split_name}2017_tax.json")
        save_coco_json(data, tax_path)
        processed_datasets_taxonomic[split_name] = data # Store the cleaned data

        # Generate and save agnostic version
        agnostic_data = generate_agnostic_labels_for_dataset(data, args.target_category_name)
        agnostic_path = os.path.join(args.final_output_coco_dir, "annotations", f"instances_{split_name}2017.json")
        save_coco_json(agnostic_data, agnostic_path)
        processed_datasets_agnostic[split_name] = agnostic_data

    # 10. Copy images to final train/val/test folders
    if not args.skip_final_image_copy:
        print("\nCopying images to final split directories...")
        # Use the taxonomic (or agnostic, image list is same) data for image list
        for split_name, data_for_split in processed_datasets_taxonomic.items(): 
            copy_images_to_final_coco_split_folder(
                data_for_split, 
                split_name, 
                args.content_root_dir, # Root for finding original bulk .jpgs
                args.final_output_coco_dir
            )
    else:
        print("\nSkipping final image copy as per --skip_final_image_copy flag.")

    print("\nScript finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tiled COCO annotations, project to bulk images, and split into train/val/test sets.")

    # --- Input Files and Paths ---
    parser.add_argument('--input_coco_json', type=str, required=True,
                        help="Path to the initial COCO JSON file with tiled annotations.")
    parser.add_argument('--content_root_dir', type=str, default="data",
                        help="Root directory where image data (e.g., 'batch-1', 'batch-2', 'bulk_batch_1_and_2') is located. Paths in COCO JSON and for metadata will be relative to this.")
    parser.add_argument('--batch1_data_dir_name', type=str, default="batch-1",
                        help="Name of the directory for batch-1 data (containing image subfolders with tiles and metadata), relative to --content_root_dir.")
    parser.add_argument('--batch2_data_dir_name', type=str, default="batch-2",
                        help="Name of the directory for batch-2 data, relative to --content_root_dir.")
    parser.add_argument('--bulk_images_dir_name', type=str, default="bulk_batch_1_and_2",
                        help="Name of the directory containing the original (non-tiled) bulk .jpg images, relative to --content_root_dir. Used for plotting and final COCO image dimensions/copying.")

    # --- Output Paths ---
    parser.add_argument('--output_dir_processed_data', type=str, required=True,
                        help="Directory to save intermediate processed data, like plotted images and the assembled (pre-split) COCO JSON.")
    parser.add_argument('--final_output_coco_dir', type=str, required=True,
                        help="Base directory for the final COCO dataset output structure (e.g., .../lifeplan_b_v9), containing annotations/ and train2017/, etc.")

    # --- Processing Parameters ---
    parser.add_argument('--target_category_name', type=str, default="b",
                        help="Target category name to filter annotations by (e.g., 'b' for bugs). Set to empty string or omit to skip filtering by name.")
    parser.add_argument('--min_shifted_area', type=int, default=15,
                        help="Minimum area (pixels squared) for a shifted annotation to be kept.")
    parser.add_argument('--exclusion_list', type=str, default="",
                        help="Comma-separated string of bulk image names (without .jpg) to exclude from processing.")
    
    # --- Splitting Parameters ---
    parser.add_argument('--val_prefixes_list', type=str, default="GYTL2T,GUSA7Z,GRLLN6",
                        help="Comma-separated list of image filename prefixes for the validation set.")
    parser.add_argument('--test_prefixes_list', type=str, default="G9XHGJ,G5JHXH,GGCNMZ,GZDUDR,GEB17N,GPRD1F",
                        help="Comma-separated list of image filename prefixes for the test set.")

    # --- Control Flags ---
    parser.add_argument('--plot_intermediate_annotations', action='store_true',
                        help="Flag to plot projected annotations on bulk images during processing.")
    parser.add_argument('--save_intermediate_images', action='store_true',
                        help="Flag to save the plotted intermediate images (if plotting is also enabled or implicitly by this).")
    parser.add_argument('--skip_final_image_copy', action='store_true',
                        help="Flag to skip copying images to the final train/val/test2017 folders.")
    parser.add_argument('--expected_annotated_images', type=int, default=49,
                        help="Expected number of final annotated bulk images after processing. Used for a sanity check. Set to 0 to disable check.")

    args = parser.parse_args()
    main(args)