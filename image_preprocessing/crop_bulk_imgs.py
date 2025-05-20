import json
import os
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

SAVE_PATH = "/content/drive/MyDrive/LIFEPLAN_Annotated_Bulk_Imgs/Revised_Images_b_Mar_2025"
base_dir = "/content/drive/MyDrive/lifeplan_b_v9"

with open(f"{SAVE_PATH}/coco_format_assembled_b_v9.json", "r") as f:
  assembled_coco_data = json.load(f)

# Iterate through each train2017 and val2017 JSON
with open(f"{base_dir}/annotations/instances_train2017.json", "r") as file:
    train_data = json.load(file)
with open(f"{base_dir}/annotations/instances_val2017.json", "r") as file:
    val_data = json.load(file)
with open(f"{base_dir}/annotations/instances_test2017.json", "r") as file:
    test_data = json.load(file)

print(len(train_data['annotations']), len(val_data['annotations']), len(test_data['annotations']))

def calculate_image_bounding_box(annotations, image_width, image_height, buffer_pixels=0):
    """Calculates the bounding box for all annotations in an image with optional buffer.

    Args:
        annotations: A list of annotation dictionaries for a single image.
        image_width: The width of the image.
        image_height: The height of the image.
        buffer_pixels: Number of pixels to add as buffer around the bounding box.

    Returns:
        A tuple (min_x, min_y, max_x, max_y) representing the bounding box,
        or None if there are no valid annotations.
    """
    min_x = image_width
    min_y = image_height
    max_x = 0
    max_y = 0
    has_valid_annotation = False

    for annotation in annotations:
        if 'segmentation' in annotation and annotation['segmentation']:
            if isinstance(annotation['segmentation'], list):
                for polygon in annotation['segmentation']:
                    if len(polygon) % 2 != 0:
                        continue  # Skip if polygon is invalid
                    x_coords = polygon[::2]
                    y_coords = polygon[1::2]

                    min_x = min(min(x_coords), min_x)
                    min_y = min(min(y_coords), min_y)
                    max_x = max(max(x_coords), max_x)
                    max_y = max(max(y_coords), max_y)
                    has_valid_annotation = True

    if has_valid_annotation:
        # Apply buffer while ensuring bounds
        min_x = max(0, min_x - buffer_pixels)
        min_y = max(0, min_y - buffer_pixels)
        max_x = min(image_width, max_x + buffer_pixels)
        max_y = min(image_height, max_y + buffer_pixels)
        return min_x, min_y, max_x, max_y
    else:
        return None

def visualize_and_save_bounding_box(image_path, annotations, bbox, output_dir):
    """Visualizes bounding box and masks on an image using OpenCV and saves the result.

    Args:
        image_path: Path to the image file.
        annotations: A list of annotation dictionaries for a single image.
        bbox: A tuple (min_x, min_y, max_x, max_y) representing the bounding box.
        output_dir: Directory where the visualized image will be saved.
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return

        # Create copy for visualization
        viz_img = img.copy()

        # # Draw masks
        # for annotation in annotations:
        #     if 'segmentation' in annotation and annotation['segmentation']:
        #         if isinstance(annotation['segmentation'], list):
        #             for polygon in annotation['segmentation']:
        #                 if len(polygon) % 2 != 0:
        #                     continue

        #                 # Convert polygon coordinates to numpy array
        #                 points = np.array(list(zip(polygon[::2], polygon[1::2])), dtype=np.int32)

        #                 # Draw filled polygon as mask
        #                 cv2.fillPoly(viz_img, [points], color=(97, 73, 164), lineType=cv2.LINE_AA)

        #                 # Draw polygon outline
        #                 cv2.polylines(viz_img, [points], True, color=(97, 73, 164),
        #                             thickness=2, lineType=cv2.LINE_AA)

        # Draw the bounding box
        if bbox:
            min_x, min_y, max_x, max_y = [int(coord) for coord in bbox]
            cv2.rectangle(viz_img, (min_x, min_y), (max_x, max_y),
                         color=(139, 0, 0), thickness=3, lineType=cv2.LINE_AA)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the visualized image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, viz_img)
        print(f"Saved visualization to: {output_path}")

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

def process_dataset(annotations_file, base_dir, split, output_dir, buffer_pixels=0):
    """Process the dataset to get bounding boxes for each image.

    Args:
        annotations_file: Path to the COCO format annotations file.
        base_dir: Base directory containing the dataset.
        split: Dataset split ('train', 'val', or 'test').
        output_dir: Directory where visualized images will be saved.
        buffer_pixels: Number of pixels to add as buffer around bounding boxes.

    Returns:
        Dictionary mapping image names to their bounding box coordinates.
    """
    # Load COCO annotations
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {annotations_file}")
        return {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in annotations file")
        return {}

    bounding_boxes = {}

    # Create split-specific output directory
    split_output_dir = os.path.join(output_dir, f"{split}2017")
    os.makedirs(split_output_dir, exist_ok=True)

    # Process each image
    for image_data in data['images']:
        image_id = image_data['id']
        image_name = image_data['file_name']
        image_width = image_data['width']
        image_height = image_data['height']

        # Get annotations for the current image
        image_annotations = [ann for ann in data['annotations']
                           if ann['image_id'] == image_id]

        # Calculate bounding box for this image
        bbox = calculate_image_bounding_box(
            image_annotations,
            image_width,
            image_height,
            buffer_pixels
        )

        if bbox:
            bounding_boxes[image_name] = bbox

            # Visualize and save the bounding box
            image_path = os.path.join(base_dir, f"{split}2017", image_name)
            print(f"Processing: {image_path}")
            visualize_and_save_bounding_box(image_path, image_annotations, bbox, split_output_dir)
        else:
            print(f"No valid annotations for image: {image_name}")

    return bounding_boxes

# Set your paths here
base_dir = "/content/drive/MyDrive/lifeplan_b_v9"
output_dir = "/content/drive/MyDrive/lifeplan_b_v9_crop_visualizations_on_original"  # Specify your output directory
splits = ['train', 'val', 'test']
buffer_pixels = 200  # Set your desired buffer size here

manual_cropped_boxes = []
for split in splits:
    annotations_file = os.path.join(base_dir, f"annotations/instances_{split}2017.json")
    # Process the dataset and get bounding boxes with buffer
    bounding_boxes = process_dataset(annotations_file, base_dir, split, output_dir, buffer_pixels)
    manual_cropped_boxes.append(bounding_boxes)

    # Print the results
    print(f"\nBounding boxes for {split} split:")
    for image_name, bbox in bounding_boxes.items():
        print(f"{image_name}: {bbox}")

manual_cropped_dict = {}
for d in manual_cropped_boxes:
  for k, v in d.items():
    manual_cropped_dict[k] = v

class COCODatasetCropper:
    def __init__(self, base_dir):
        """Initialize the COCO dataset cropper.

        Args:
            base_dir: Base directory containing the COCO dataset
        """
        self.base_dir = base_dir
        self.splits = ['train', 'val', 'test']
        self.year = '2017'  # Change if using different COCO version

    def load_annotations(self, split):
        """Load annotation file for given split.

        Args:
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Loaded annotation data or None if file doesn't exist
        """
        ann_file = os.path.join(
            self.base_dir,
            'annotations',
            f'instances_{split}{self.year}.json'
        )
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                return json.load(f)
        return None

    def validate_image_bbox(self, bbox, image_annotations):
        """Validate that the bounding box includes all annotations for a specific image.

        Args:
            bbox: Tuple (min_x, min_y, max_x, max_y)
            image_annotations: List of annotations for the image

        Returns:
            Boolean indicating if bbox is valid and list of any truncated annotations
        """
        min_x, min_y, max_x, max_y = bbox
        truncated_annotations = []

        for ann in image_annotations:
            if 'segmentation' in ann and ann['segmentation']:
                if isinstance(ann['segmentation'], list):
                    for polygon in ann['segmentation']:
                        if len(polygon) % 2 != 0:
                            continue
                        x_coords = polygon[::2]
                        y_coords = polygon[1::2]

                        # Check if any point lies outside the bbox
                        if (any(x < min_x or x > max_x for x in x_coords) or
                            any(y < min_y or y > max_y for y in y_coords)):
                            truncated_annotations.append(ann['id'])

        return len(truncated_annotations) == 0, truncated_annotations

    def adjust_image_coordinates(self, image_annotations, bbox):
        """Adjust annotation coordinates for a specific image based on its cropping bbox.

        Args:
            image_annotations: List of annotations for the image
            bbox: Tuple (min_x, min_y, max_x, max_y)

        Returns:
            List of adjusted annotations
        """
        min_x, min_y, max_x, max_y = bbox
        adjusted_annotations = []

        for ann in image_annotations:
            adjusted_ann = ann.copy()
            if 'segmentation' in adjusted_ann and adjusted_ann['segmentation']:
                if isinstance(adjusted_ann['segmentation'], list):
                    adjusted_segments = []
                    for polygon in adjusted_ann['segmentation']:
                        if len(polygon) % 2 != 0:
                            continue
                        adjusted_polygon = []
                        for i in range(0, len(polygon), 2):
                            # Adjust x coordinate
                            adj_x = polygon[i] - min_x
                            # Adjust y coordinate
                            adj_y = polygon[i + 1] - min_y
                            adjusted_polygon.extend([adj_x, adj_y])
                        adjusted_segments.append(adjusted_polygon)
                    adjusted_ann['segmentation'] = adjusted_segments

                    # Recalculate bbox based on adjusted segmentation
                    x_coords = [p for polygon in adjusted_segments for p in polygon[::2]]
                    y_coords = [p for polygon in adjusted_segments for p in polygon[1::2]]
                    if x_coords and y_coords:
                        adjusted_ann['bbox'] = [
                            min(x_coords),
                            min(y_coords),
                            max(x_coords) - min(x_coords),
                            max(y_coords) - min(y_coords)
                        ]
            adjusted_annotations.append(adjusted_ann)

        return adjusted_annotations

    def visualize_annotations(self, image_path, annotations, output_path=None):
        """Visualize annotations on an image using OpenCV.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries
            output_path: Optional path to save visualization
        """
        try:
            import cv2
            import numpy as np

            # Read image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Create two masks: one for fill and one for borders
            mask_fill = np.zeros(img.shape[:2], dtype=np.uint8)
            mask_border = np.zeros(img.shape[:2], dtype=np.uint8)

            # Fill in all annotation polygons
            for ann in annotations:
                if 'segmentation' in ann and ann['segmentation']:
                    if isinstance(ann['segmentation'], list):
                        for polygon in ann['segmentation']:
                            if len(polygon) % 2 != 0:
                                continue

                            # Convert polygon to numpy array of points
                            points = np.array(list(zip(polygon[::2], polygon[1::2])), dtype=np.int32)

                            # Fill polygon in the fill mask
                            cv2.fillPoly(mask_fill, [points], 255)

                            # Draw polygon border in the border mask
                            cv2.polylines(mask_border, [points], True, 255, thickness=2)

            # Create colored overlays
            overlay = img.copy()
            # Blue fill (BGR format: 255, 127, 0 is a medium blue)
            overlay[mask_fill > 0] = [255, 127, 0]
            # Darker blue border (BGR format: 255, 0, 0 is a darker blue)
            overlay[mask_border > 0] = [255, 0, 0]

            # Blend the original image and overlay
            alpha = 0.4
            output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Add the dark borders on top with full opacity
            output[mask_border > 0] = [255, 0, 0]

            if output_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Save the visualization
                cv2.imwrite(output_path, output)

            return output

        except Exception as e:
            print(f"Error visualizing annotations: {str(e)}")
            return None

    def process_split(self, split, bboxes_dict, output_base_dir, visualize=False):
        """Process a single dataset split.

        Args:
            split: Dataset split name ('train', 'val', or 'test')
            bboxes_dict: Dictionary mapping image names to their bounding boxes
            output_base_dir: Base directory for output
            visualize: Whether to create visualization of annotations

        Returns:
            Boolean indicating success
        """
        print(f"\nProcessing {split} split...")

        # Load annotations
        annotations_data = self.load_annotations(split)
        if not annotations_data:
            print(f"No annotations found for {split} split")
            return False

        # Setup input/output directories
        input_dir = os.path.join(self.base_dir, f'{split}{self.year}')
        output_dir = os.path.join(output_base_dir, f'{split}{self.year}')
        os.makedirs(output_dir, exist_ok=True)

        # Create new annotations structure
        adjusted_data = annotations_data.copy()
        adjusted_data['annotations'] = []
        adjusted_data['images'] = []

        # Process images
        print("Cropping images...")
        success_count = 0

        for img in tqdm(annotations_data['images']):
            if img['file_name'] not in bboxes_dict:
                print(f"Warning: No bounding box found for {img['file_name']}")
                continue

            bbox = bboxes_dict[img['file_name']]

            # Get annotations for this image
            img_annotations = [
                ann for ann in annotations_data['annotations']
                if ann['image_id'] == img['id']
            ]

            # Validate bbox
            is_valid, truncated = self.validate_image_bbox(bbox, img_annotations)
            if not is_valid:
                print(f"Warning: Bounding box would truncate annotations in {img['file_name']}")
                continue

            # Crop and save image, then get actual dimensions
            input_path = os.path.join(input_dir, img['file_name'])
            output_path = os.path.join(output_dir, img['file_name'])
            output_path = output_path.replace(".jpg", ".png")

            try:
                with Image.open(input_path) as image:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    min_x, min_y, max_x, max_y = bbox
                    cropped = image.crop((min_x, min_y, max_x, max_y))

                    # Get actual dimensions of cropped image    # NEW
                    actual_width, actual_height = cropped.size  # NEW

                    # Create output directory if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Save the cropped image
                    cropped.save(output_path, format='PNG')

                    # Adjust image dimensions using actual cropped size
                    adjusted_img = img.copy()
                    adjusted_img['width'] = actual_width      # Changed: Uses actual size
                    adjusted_img['height'] = actual_height    # Changed: Uses actual size
                    adjusted_img['file_name'] = img['file_name'].replace('.jpg', '.png')
                    adjusted_img['path'] = img['path'].replace('.jpg', '.png')
                    adjusted_data['images'].append(adjusted_img)

                    # Adjust annotations
                    adjusted_annotations = self.adjust_image_coordinates(img_annotations, bbox)
                    adjusted_data['annotations'].extend(adjusted_annotations)

                    success_count += 1

                    # Visualize if requested
                    if visualize:
                        vis_dir = os.path.join(output_base_dir, f'{split}{self.year}_visualizations')
                        os.makedirs(vis_dir, exist_ok=True)
                        output_vis_path = os.path.join(
                            vis_dir,
                            f"{os.path.splitext(img['file_name'])[0]}_annotated.png"
                        )
                        self.visualize_annotations(output_path, adjusted_annotations, output_vis_path)

            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                continue

        # Save adjusted annotations
        output_ann_dir = os.path.join(output_base_dir, 'annotations')
        os.makedirs(output_ann_dir, exist_ok=True)
        with open(os.path.join(output_ann_dir, f'instances_{split}{self.year}.json'), 'w') as f:
            json.dump(adjusted_data, f)

        print(f"Successfully processed {success_count}/{len(annotations_data['images'])} images")
        return True

    def process_dataset(self, bboxes_dict, output_base_dir, visualize=False):
        """Process entire dataset across all splits.

        Args:
            bboxes_dict: Dictionary mapping image names to their bounding boxes
            output_base_dir: Base directory for output
            visualize: Whether to create visualization of annotations

        Returns:
            Boolean indicating overall success
        """
        print(f"Processing dataset with {len(bboxes_dict)} image bounding boxes")
        print(f"Output directory: {output_base_dir}")

        success = True
        for split in self.splits:
            split_success = self.process_split(split, bboxes_dict, output_base_dir, visualize)
            success = success and split_success

        return success

# Example usage
base_dir = "/content/drive/MyDrive/lifeplan_b_v9"
output_dir = "/content/drive/MyDrive/lifeplan_b_v9_cropped_png"

# Example bounding boxes dictionary
bboxes_dict = manual_cropped_dict

# Initialize and run the cropper
cropper = COCODatasetCropper(base_dir)
success = cropper.process_dataset(bboxes_dict, output_dir, visualize=True)

if success:
    print("\nDataset processing completed successfully!")
    print(f"Cropped images and annotations saved to: {output_dir}")
    print(f"Visualizations saved to: {output_dir}/<split>_visualizations/")
else:
    print("\nErrors occurred during dataset processing.")


