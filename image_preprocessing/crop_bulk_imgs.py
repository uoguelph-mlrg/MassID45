import json
import os
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import argparse

def calculate_image_bounding_box(annotations, image_width, image_height, buffer_pixels=0):
    """Calculates the bounding box for all annotations in an image with optional buffer."""
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    has_valid_annotation = False

    for annotation in annotations:
        if 'segmentation' in annotation and annotation['segmentation']:
            # Ensure segmentation is a list of polygons
            segments = annotation['segmentation']
            if not isinstance(segments, list):
                # If it's a single RLE or polygon, wrap it in a list
                segments = [segments]
            
            for polygon_coords in segments:
                # Handle RLE (not fully implemented here, assumes polygon list)
                if isinstance(polygon_coords, dict) and 'counts' in polygon_coords:
                    # print(f"Warning: RLE segmentation found for annotation {annotation.get('id')}, bounding box from RLE not implemented. Using annotation bbox if available.")
                    if 'bbox' in annotation:
                        ann_x, ann_y, ann_w, ann_h = annotation['bbox']
                        min_x = min(ann_x, min_x)
                        min_y = min(ann_y, min_y)
                        max_x = max(ann_x + ann_w, max_x)
                        max_y = max(ann_y + ann_h, max_y)
                        has_valid_annotation = True
                    continue

                # Assuming polygon_coords is a flat list [x1, y1, x2, y2, ...]
                if not isinstance(polygon_coords, list) or len(polygon_coords) < 6 or len(polygon_coords) % 2 != 0:
                    # print(f"Warning: Invalid polygon format or too few points for annotation {annotation.get('id')}. Length: {len(polygon_coords) if isinstance(polygon_coords, list) else 'Not a list'}")
                    continue 
                
                x_coords = polygon_coords[::2]
                y_coords = polygon_coords[1::2]

                if not x_coords or not y_coords: continue # Should not happen with len check

                current_min_x = min(x_coords)
                current_min_y = min(y_coords)
                current_max_x = max(x_coords)
                current_max_y = max(y_coords)
                
                min_x = min(current_min_x, min_x)
                min_y = min(current_min_y, min_y)
                max_x = max(current_max_x, max_x)
                max_y = max(current_max_y, max_y)
                has_valid_annotation = True
    
    if not has_valid_annotation and min_x == float('inf'): # No valid segmentations found
        # print("No valid segmentations to calculate bounding box.")
        return None

    # Apply buffer while ensuring bounds
    min_x = max(0, min_x - buffer_pixels)
    min_y = max(0, min_y - buffer_pixels)
    max_x = min(image_width, max_x + buffer_pixels)
    max_y = min(image_height, max_y + buffer_pixels)
    
    # Ensure min is not greater than max (can happen if buffer is too large or initial box is tiny)
    if min_x >= max_x or min_y >= max_y:
        # print(f"Warning: Invalid bounding box after buffer: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}. Reverting to no buffer or original image size if no annots.")
        # Fallback logic: try without buffer, or if still invalid, might return None or full image
        # This part might need more sophisticated handling based on requirements.
        # For now, let's re-calculate without buffer if this happens.
        if buffer_pixels > 0:
            return calculate_image_bounding_box(annotations, image_width, image_height, 0) # Try no buffer
        else: # Already no buffer, box is inherently invalid (e.g. all points are the same)
            return None


    return min_x, min_y, max_x, max_y

def visualize_and_save_bounding_box(image_path, annotations, bbox, output_dir, plot_masks=False):
    """Visualizes bounding box and masks on an image using OpenCV and saves the result."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return

        viz_img = img.copy()

        if plot_masks:
            for annotation in annotations:
                if 'segmentation' in annotation and annotation['segmentation']:
                    segments = annotation['segmentation']
                    if not isinstance(segments, list): segments = [segments] # Ensure list

                    for polygon_coords in segments:
                        if isinstance(polygon_coords, dict): continue # Skip RLE for mask plotting here
                        if not isinstance(polygon_coords, list) or len(polygon_coords) < 6 or len(polygon_coords) % 2 != 0:
                            continue
                        
                        points = np.array(list(zip(polygon_coords[::2], polygon_coords[1::2])), dtype=np.int32)
                        cv2.fillPoly(viz_img, [points], color=(97, 73, 164), lineType=cv2.LINE_AA)
                        cv2.polylines(viz_img, [points], True, color=(97, 73, 164), thickness=2, lineType=cv2.LINE_AA)

        if bbox:
            min_x, min_y, max_x, max_y = [int(coord) for coord in bbox]
            cv2.rectangle(viz_img, (min_x, min_y), (max_x, max_y), color=(139, 0, 0), thickness=3, lineType=cv2.LINE_AA)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, viz_img)
        # print(f"Saved BBox visualization to: {output_path}")

    except Exception as e:
        print(f"Error processing image {image_path} for BBox visualization: {str(e)}")

def generate_bounding_boxes_for_dataset(annotations_file_path, input_image_dir_for_split, 
                                       visualization_output_dir_for_split, 
                                       buffer_pixels=0, plot_masks_on_viz=False):
    """Process the dataset to get bounding boxes for each image."""
    try:
        with open(annotations_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {annotations_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in annotations file {annotations_file_path}")
        return {}

    image_name_to_bbox_map = {}
    if visualization_output_dir_for_split:
        os.makedirs(visualization_output_dir_for_split, exist_ok=True)

    print(f"Generating bounding boxes for images in {input_image_dir_for_split}...")
    for image_data in tqdm(data.get('images', []), desc="Generating BBoxes"):
        image_id = image_data['id']
        image_name = image_data['file_name']
        image_width = image_data.get('width')
        image_height = image_data.get('height')

        if image_width is None or image_height is None:
            # print(f"Warning: Missing width/height for image {image_name}. Skipping.")
            # Optionally, try to read from disk if path is available, but COCO should have it.
            # For now, skip if not in JSON.
            img_path_check = os.path.join(input_image_dir_for_split, image_name)
            if os.path.exists(img_path_check):
                try:
                    img_temp = Image.open(img_path_check)
                    image_width, image_height = img_temp.size
                    # print(f"  Read dimensions from disk for {image_name}: {image_width}x{image_height}")
                except Exception as e_read:
                    print(f"  Could not read dimensions from disk for {image_name}: {e_read}. Skipping bbox calculation.")
                    continue
            else:
                print(f"Warning: Image {image_name} not found at {img_path_check} and dimensions missing in JSON. Skipping bbox calculation.")
                continue


        image_annotations = [ann for ann in data.get('annotations', []) if ann['image_id'] == image_id]

        bbox = calculate_image_bounding_box(image_annotations, image_width, image_height, buffer_pixels)

        if bbox:
            image_name_to_bbox_map[image_name] = bbox
            if visualization_output_dir_for_split:
                image_full_path = os.path.join(input_image_dir_for_split, image_name)
                visualize_and_save_bounding_box(image_full_path, image_annotations, bbox, 
                                                visualization_output_dir_for_split, plot_masks_on_viz)
        # else:
            # print(f"No valid annotations or bounding box could be calculated for image: {image_name}")
            
    return image_name_to_bbox_map

class COCODatasetCropper:
    def __init__(self, input_coco_base_dir):
        self.input_coco_base_dir = input_coco_base_dir

    def _load_annotations(self, split):
        ann_file = os.path.join(self.input_coco_base_dir, 'annotations', f'instances_{split}2017.json')
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                return json.load(f)
        print(f"Warning: Annotation file not found for {split} at {ann_file}")
        return None

    def _validate_image_bbox_contains_all_segmentations(self, bbox_coords, image_annotations_list):
        min_x_crop, min_y_crop, max_x_crop, max_y_crop = bbox_coords
        truncated_annotation_ids = []

        for ann in image_annotations_list:
            if 'segmentation' in ann and ann['segmentation']:
                segments = ann['segmentation']
                if not isinstance(segments, list): segments = [segments]

                for polygon_coords in segments:
                    if isinstance(polygon_coords, dict): continue # Skip RLE for this validation
                    if not isinstance(polygon_coords, list) or len(polygon_coords) < 6 or len(polygon_coords) % 2 != 0:
                        continue
                    
                    x_coords = polygon_coords[::2]
                    y_coords = polygon_coords[1::2]

                    if (any(x < min_x_crop or x > max_x_crop for x in x_coords) or
                        any(y < min_y_crop or y > max_y_crop for y in y_coords)):
                        truncated_annotation_ids.append(ann['id'])
                        break # This annotation is truncated, no need to check its other segments
        
        return not truncated_annotation_ids, truncated_annotation_ids # True if no truncations

    def _adjust_annotation_coords_to_cropped_image(self, image_annotations_list, crop_bbox_coords):
        crop_origin_x, crop_origin_y, _, _ = crop_bbox_coords
        adjusted_annotations_list = []

        for ann_orig in image_annotations_list:
            adj_ann = copy.deepcopy(ann_orig) # Work on a copy
            if 'segmentation' in adj_ann and adj_ann['segmentation']:
                original_segments = adj_ann['segmentation']
                if not isinstance(original_segments, list): original_segments = [original_segments]
                
                adjusted_segments_for_this_ann = []
                all_adj_x_coords_for_ann = []
                all_adj_y_coords_for_ann = []

                for polygon_coords in original_segments:
                    if isinstance(polygon_coords, dict): # RLE
                        # Adjusting RLE is complex, often involves re-encoding.
                        # For now, if original was RLE, we might keep it as RLE but note that its relation to image changes.
                        # Or, convert to polygon, adjust, then decide if re-encoding is needed.
                        # Simplest: skip RLE adjustment or copy as is. Let's copy as is and rely on bbox.
                        # print(f"Warning: RLE segmentation for ann {adj_ann['id']} in adjusted data. Positional accuracy relative to crop needs care.")
                        adjusted_segments_for_this_ann.append(polygon_coords) # Copy RLE as is
                        # Try to use original bbox for RLE if needed for adjusted_ann['bbox']
                        if 'bbox' in ann_orig:
                             rle_x, rle_y, rle_w, rle_h = ann_orig['bbox']
                             all_adj_x_coords_for_ann.extend([rle_x - crop_origin_x, rle_x - crop_origin_x + rle_w])
                             all_adj_y_coords_for_ann.extend([rle_y - crop_origin_y, rle_y - crop_origin_y + rle_h])
                        continue

                    if not isinstance(polygon_coords, list) or len(polygon_coords) < 6 or len(polygon_coords) % 2 != 0:
                        adjusted_segments_for_this_ann.append(polygon_coords) # Copy invalid as is
                        continue 

                    adj_poly = []
                    current_poly_adj_x = []
                    current_poly_adj_y = []
                    for i in range(0, len(polygon_coords), 2):
                        adj_x = polygon_coords[i] - crop_origin_x
                        adj_y = polygon_coords[i+1] - crop_origin_y
                        adj_poly.extend([adj_x, adj_y])
                        current_poly_adj_x.append(adj_x)
                        current_poly_adj_y.append(adj_y)
                    
                    adjusted_segments_for_this_ann.append(adj_poly)
                    all_adj_x_coords_for_ann.extend(current_poly_adj_x)
                    all_adj_y_coords_for_ann.extend(current_poly_adj_y)
                
                adj_ann['segmentation'] = adjusted_segments_for_this_ann

                # Recalculate bbox for the annotation based on all its adjusted segments
                if all_adj_x_coords_for_ann and all_adj_y_coords_for_ann:
                    min_overall_x = min(all_adj_x_coords_for_ann)
                    min_overall_y = min(all_adj_y_coords_for_ann)
                    max_overall_x = max(all_adj_x_coords_for_ann)
                    max_overall_y = max(all_adj_y_coords_for_ann)
                    adj_ann['bbox'] = [
                        min_overall_x,
                        min_overall_y,
                        max_overall_x - min_overall_x,
                        max_overall_y - min_overall_y
                    ]
                elif 'bbox' in adj_ann: # If no valid segmentations to derive bbox, remove old one
                    del adj_ann['bbox']


            adjusted_annotations_list.append(adj_ann)
        return adjusted_annotations_list

    def _visualize_adjusted_annotations(self, cropped_image_path, adjusted_annotations_list, visualization_output_path):
        try:
            img = cv2.imread(cropped_image_path)
            if img is None:
                raise ValueError(f"Could not read cropped image for visualization: {cropped_image_path}")

            # Colors for visualization (adapt as needed)
            fill_color_bgr = (255, 127, 0)  # Medium blue for fill
            border_color_bgr = (255, 0, 0)   # Darker blue for border
            
            # Create a copy for drawing
            viz_img = img.copy()
            
            # Create overlays for fill and border to handle transparency better
            overlay_fill = np.zeros_like(viz_img, dtype=np.uint8)
            overlay_border = np.zeros_like(viz_img, dtype=np.uint8)


            for ann in adjusted_annotations_list:
                if 'segmentation' in ann and ann['segmentation']:
                    segments = ann['segmentation']
                    if not isinstance(segments, list): segments = [segments]

                    for polygon_coords in segments:
                        if isinstance(polygon_coords, dict): continue # Skip RLE
                        if not isinstance(polygon_coords, list) or len(polygon_coords) < 6 or len(polygon_coords) % 2 != 0:
                            continue
                        
                        points = np.array(list(zip(polygon_coords[::2], polygon_coords[1::2])), dtype=np.int32)
                        
                        cv2.fillPoly(overlay_fill, [points], fill_color_bgr)
                        cv2.polylines(overlay_border, [points], True, border_color_bgr, thickness=2)
            
            # Blend fill overlay
            alpha = 0.4 # Transparency for fill
            cv2.addWeighted(overlay_fill, alpha, viz_img, 1 - alpha, 0, viz_img)
            
            # Add borders on top (no transparency)
            viz_img[overlay_border[:,:,0] == border_color_bgr[0]] = border_color_bgr # A bit hacky, better way is per-pixel


            os.makedirs(os.path.dirname(visualization_output_path), exist_ok=True)
            cv2.imwrite(visualization_output_path, viz_img)
            # print(f"Saved annotation visualization to: {visualization_output_path}")

        except Exception as e:
            print(f"Error visualizing adjusted annotations for {cropped_image_path}: {str(e)}")


    def process_split(self, split_name, image_to_bbox_map, output_dataset_base_dir, create_visualizations=False):
        print(f"\nProcessing COCO split: {split_name}")
        input_annotations_data = self._load_annotations(split_name)
        if not input_annotations_data: return False

        input_image_dir_for_split = os.path.join(self.input_coco_base_dir, f'{split_name}2017')
        output_image_dir_for_split = os.path.join(output_dataset_base_dir, f'{split_name}2017')
        os.makedirs(output_image_dir_for_split, exist_ok=True)

        # Prepare structure for the new (cropped) COCO annotation file
        output_annotations_data = copy.deepcopy(input_annotations_data)
        output_annotations_data['annotations'] = []
        output_annotations_data['images'] = []

        successful_crops = 0
        for img_entry_orig in tqdm(input_annotations_data.get('images',[]), desc=f"Cropping {split_name} images"):
            image_filename = img_entry_orig['file_name']
            
            if image_filename not in image_to_bbox_map:
                # print(f"Warning: No bounding box provided for image {image_filename}. Skipping this image.")
                continue
            
            crop_bbox = image_to_bbox_map[image_filename] # (min_x, min_y, max_x, max_y)

            annotations_for_this_image = [
                ann for ann in input_annotations_data.get('annotations',[]) 
                if ann['image_id'] == img_entry_orig['id']
            ]

            # Optional: Validate that the crop_bbox doesn't truncate any critical part of annotations
            # is_bbox_valid, _ = self._validate_image_bbox_contains_all_segmentations(crop_bbox, annotations_for_this_image)
            # if not is_bbox_valid:
            #     print(f"Warning: Crop bounding box for {image_filename} would truncate annotations. Skipping this image.")
            #     continue # Or handle differently, e.g., adjust bbox

            input_image_path = os.path.join(input_image_dir_for_split, image_filename)
            # Output images will be PNG
            output_image_filename_png = os.path.splitext(image_filename)[0] + ".png"
            output_image_path_png = os.path.join(output_image_dir_for_split, output_image_filename_png)

            try:
                with Image.open(input_image_path) as img_pil:
                    if img_pil.mode != 'RGB': img_pil = img_pil.convert('RGB')
                    
                    min_x, min_y, max_x, max_y = map(int, crop_bbox) # Ensure integer coordinates for PIL crop
                    cropped_img_pil = img_pil.crop((min_x, min_y, max_x, max_y))
                    cropped_img_pil.save(output_image_path_png, format='PNG')

                actual_cropped_width, actual_cropped_height = cropped_img_pil.size

                # Update image entry for the new COCO JSON
                adj_img_entry = copy.deepcopy(img_entry_orig)
                adj_img_entry['width'] = actual_cropped_width
                adj_img_entry['height'] = actual_cropped_height
                adj_img_entry['file_name'] = output_image_filename_png
                if 'path' in adj_img_entry: # If original had 'path', update it
                     adj_img_entry['path'] = os.path.join(os.path.dirname(adj_img_entry.get('path', '')), output_image_filename_png)

                output_annotations_data['images'].append(adj_img_entry)

                # Adjust annotation coordinates
                adjusted_annots = self._adjust_annotation_coords_to_cropped_image(annotations_for_this_image, crop_bbox)
                output_annotations_data['annotations'].extend(adjusted_annots)
                
                successful_crops += 1

                if create_visualizations:
                    vis_dir = os.path.join(output_dataset_base_dir, f'{split_name}2017_annot_visualizations')
                    output_vis_path = os.path.join(vis_dir, f"{os.path.splitext(output_image_filename_png)[0]}_annot.png")
                    self._visualize_adjusted_annotations(output_image_path_png, adjusted_annots, output_vis_path)

            except FileNotFoundError:
                print(f"Error: Input image not found at {input_image_path}")
            except Exception as e:
                print(f"Error cropping/processing image {input_image_path}: {e}")
        
        # Save the new annotation file for the cropped dataset
        output_annotations_dir = os.path.join(output_dataset_base_dir, 'annotations')
        os.makedirs(output_annotations_dir, exist_ok=True)
        output_ann_file_path = os.path.join(output_annotations_dir, f'instances_{split_name}2017.json')
        with open(output_ann_file_path, 'w') as f:
            json.dump(output_annotations_data, f, indent=2) # Using indent=2 for smaller files

        print(f"Successfully cropped {successful_crops}/{len(input_annotations_data.get('images',[]))} images for {split_name} split.")
        print(f"Cropped annotations saved to: {output_ann_file_path}")
        if create_visualizations:
            print(f"Annotation visualizations saved in: {output_dataset_base_dir}/{split_name}2017_annot_visualizations/")
        return True

    def process_entire_dataset(self, image_name_to_bbox_map, output_dataset_base_dir, create_visualizations=False):
        print(f"Starting COCO dataset cropping. Output will be in: {output_dataset_base_dir}")
        overall_success = True
        for split in ['train', 'val', 'test']: # Standard splits
            if not self.process_split(split, image_name_to_bbox_map, output_dataset_base_dir, create_visualizations):
                overall_success = False
                print(f"Processing failed for {split} split.")
        
        if overall_success:
            print("\nDataset cropping completed successfully for all splits!")
        else:
            print("\nErrors occurred during dataset cropping for one or more splits.")
        return overall_success

import copy # ensure copy is imported

def main(args):
    # --- Part 1: Generate Bounding Boxes (from original script's first part) ---
    all_splits_image_to_bbox_map = {}
    
    # Path for storing visualizations of the initially calculated bounding boxes
    bbox_visualization_base_dir = None
    if args.visualize_calculated_bboxes:
        bbox_visualization_base_dir = os.path.join(args.output_dir_cropped_dataset, "crop_visualizations")
        os.makedirs(bbox_visualization_base_dir, exist_ok=True)

    for split in args.splits_to_process:
        print(f"\n--- Generating bounding boxes for {split} split ---")
        input_annotations_file = os.path.join(args.input_base_dir, 'annotations', f'instances_{split}2017.json')
        input_image_dir = os.path.join(args.input_base_dir, f'{split}2017')
        
        split_bbox_viz_dir = None
        if bbox_visualization_base_dir:
            split_bbox_viz_dir = os.path.join(bbox_visualization_base_dir, f"{split}2017")
        
        split_bboxes = generate_bounding_boxes_for_dataset(
            annotations_file_path=input_annotations_file,
            input_image_dir_for_split=input_image_dir,
            visualization_output_dir_for_split=split_bbox_viz_dir, # Pass the specific dir for this split
            buffer_pixels=args.buffer_pixels,
            plot_masks_on_viz=args.plot_masks_on_bbox_viz 
        )
        all_splits_image_to_bbox_map.update(split_bboxes)
        
        if args.print_calculated_bboxes:
            print(f"\nCalculated Bounding boxes for {split} split (buffer: {args.buffer_pixels}px):")
            for img_name, bbox in split_bboxes.items():
                print(f"  {img_name}: {bbox}")
    
    if not all_splits_image_to_bbox_map:
        print("No bounding boxes were generated. Exiting.")
        return

    # --- Part 2: Crop Dataset using COCODatasetCropper ---
    print(f"\n--- Cropping dataset using generated bounding boxes ---")
    cropper = COCODatasetCropper(input_coco_base_dir=args.input_base_dir)
    
    success = cropper.process_entire_dataset(
        image_name_to_bbox_map=all_splits_image_to_bbox_map,
        output_dataset_base_dir=args.output_dir_cropped_dataset,
        create_visualizations=args.visualize_cropped_annotations
    )

    if success:
        print(f"\nCropping process finished. Results in: {args.output_dir_cropped_dataset}")
    else:
        print("\nCropping process encountered errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO Dataset Preprocessing: Calculate bounding boxes for annotations and crop images and annotations accordingly.")

    # --- Input Paths ---
    parser.add_argument('--input_base_dir', type=str, required=True,
                        help="Base directory of the input COCO dataset (e.g., 'lifeplan_b_v9_testing'). "
                             "Should contain 'annotations/' and split image folders like 'train2017/'.")
    
    # --- Output Paths ---
    parser.add_argument('--output_dir_cropped_dataset', type=str, required=True,
                        help="Base directory where the new cropped COCO dataset will be saved "
                             "(e.g., 'lifeplan_b_v9_testing_cropped_png'). This will also house visualizations if enabled.")

    # --- Bounding Box Calculation Parameters ---
    parser.add_argument('--buffer_pixels', type=int, default=200,
                        help="Number of pixels to add as a buffer around the calculated bounding box of all annotations in an image. Default: 200.")
    
    # --- General Parameters ---
    parser.add_argument('--splits_to_process', nargs='+', default=['train', 'val', 'test'],
                        help="List of dataset splits to process (e.g., 'train' 'val'). Default: ['train', 'val', 'test'].")

    # --- Visualization and Debugging Flags ---
    parser.add_argument('--visualize_calculated_bboxes', action='store_true',
                        help="If set, save visualizations of the initially calculated bounding boxes on the original images.")
    parser.add_argument('--plot_masks_on_bbox_viz', action='store_true',
                        help="If --visualize_calculated_bboxes is set, also plot annotation masks on these visualizations.")
    parser.add_argument('--visualize_cropped_annotations', action='store_true',
                        help="If set, save visualizations of the adjusted annotations on the newly cropped images.")
    parser.add_argument('--print_calculated_bboxes', action='store_true',
                        help="If set, print the calculated bounding box for each image to the console.")
    
    args = parser.parse_args()
    main(args)