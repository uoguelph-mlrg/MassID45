import json
import os
import shutil
from tqdm import tqdm
from sahi.slicing import slice_coco
import argparse

# Removed unused imports: numpy, matplotlib.pyplot

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
                print(f"Warning: Negative coordinate found in annotation ID {annotation.get('id', 'N/A')} image ID {annotation.get('image_id', 'N/A')}")
                # Original script breaks on first problematic annotation, then asserts.
                break
        if has_negative_coords:
            break
    assert not has_negative_coords, "Negative coordinates found in annotations."
    # This print will only be reached if the assertion passes (i.e., has_negative_coords is False)
    print(f"Has negative coordinates: {has_negative_coords}")


def get_sahi_coco_data(input_base_dir_path, crop_size_for_naming, crop_rows_val,
                       crop_cols_val, split_name, sahi_output_base_path,
                       sahi_ignore_neg_samples_flag, overlap_ratio_val,
                       min_area_ratio_val, sahi_verbose_flag):
    """
    Slices a COCO dataset split using SAHI and organizes the output.
    """
    # Determine path suffix based on ignore_negative_samples flag
    # Original script path was "_ignore_neg" but behavior was to keep negative samples.
    # This version makes the path reflect the actual flag.
    if sahi_ignore_neg_samples_flag:
        neg_sample_folder_suffix = "_ignore_neg"
    else:
        neg_sample_folder_suffix = "_keep_neg" # Negative samples are kept

    # Construct the specific SAHI dataset path for this configuration
    # e.g., /path/to/sahi_outputs/sahi_512_keep_neg
    current_sahi_dataset_path = os.path.join(sahi_output_base_path, f"sahi_{crop_size_for_naming}{neg_sample_folder_suffix}")
    print(f"Preparing SAHI dataset at: {current_sahi_dataset_path} for split: {split_name}")

    # Setup COCO dataset file structure for the output
    os.makedirs(current_sahi_dataset_path, exist_ok=True)
    
    # Directory for final sliced images for this split
    # e.g., /path/to/sahi_outputs/sahi_512_keep_neg/train2017
    final_split_image_dir = os.path.join(current_sahi_dataset_path, f"{split_name}2017")
    os.makedirs(final_split_image_dir, exist_ok=True)

    # Directory for final annotations for this SAHI configuration
    # e.g., /path/to/sahi_outputs/sahi_512_keep_neg/annotations
    final_annotations_dir = os.path.join(current_sahi_dataset_path, "annotations")
    os.makedirs(final_annotations_dir, exist_ok=True)

    # Temporary output directory for initial slicing by SAHI
    # e.g., /path/to/sahi_outputs/sahi_512_keep_neg/train2017_initial
    sahi_initial_output_dir = os.path.join(current_sahi_dataset_path, f"{split_name}2017_initial")
    # Ensure initial output dir is clean to avoid SAHI errors or old data
    if os.path.exists(sahi_initial_output_dir):
        shutil.rmtree(sahi_initial_output_dir)
    os.makedirs(sahi_initial_output_dir)

    # Paths for input data
    input_coco_annotation_file = os.path.join(input_base_dir_path, "annotations", f"instances_{split_name}2017.json")
    input_image_dir = os.path.join(input_base_dir_path, f"{split_name}2017")

    print(f"Slicing {split_name} data... Input annotation: {input_coco_annotation_file}")
    
    # Perform slicing
    # slice_coco returns:
    # - coco_dict: the dictionary of the sliced COCO data
    # - coco_path: the path to the COCO JSON file created by SAHI
    coco_dict, coco_json_path_from_sahi = slice_coco(
        coco_annotation_file_path=input_coco_annotation_file,
        image_dir=input_image_dir,
        output_coco_annotation_file_name=f"instances_{split_name}2017", # SAHI appends "_coco.json"
        ignore_negative_samples=sahi_ignore_neg_samples_flag,
        output_dir=sahi_initial_output_dir, # SAHI outputs sliced images and JSON here
        slice_height=crop_rows_val,
        slice_width=crop_cols_val,
        overlap_height_ratio=overlap_ratio_val,
        overlap_width_ratio=overlap_ratio_val,
        min_area_ratio=min_area_ratio_val,
        verbose=sahi_verbose_flag
    )
    # coco_json_path_from_sahi will be like:
    # {sahi_initial_output_dir}/instances_{split_name}2017_coco.json

    # Load the COCO data dictionary that SAHI just created (though it's also returned as coco_dict)
    with open(coco_json_path_from_sahi, "r") as f:
        sliced_coco_data = json.load(f) # This is same as coco_dict

    # Copy sliced images from SAHI's initial output to the final image directory
    # The original script's "valid_imgs" logic was essentially copying all images listed by SAHI.
    print(f"Copying {len(sliced_coco_data['images'])} sliced images from {sahi_initial_output_dir} to {final_split_image_dir}...")
    for img_entry in tqdm(sliced_coco_data["images"], desc=f"Copying images for {split_name} [{crop_size_for_naming}px]"):
        # img_entry["file_name"] is usually relative to sahi_initial_output_dir (e.g., "IMG_0_0.png" or "subdir/IMG_0_0.png")
        source_image_path = os.path.join(sahi_initial_output_dir, img_entry["file_name"])
        # Ensure flat structure in target directory by taking only the basename
        destination_image_path = os.path.join(final_split_image_dir, os.path.basename(img_entry["file_name"]))
        
        if os.path.exists(source_image_path):
            shutil.copyfile(source_image_path, destination_image_path)
        else:
            print(f"Warning: Source image {source_image_path} not found during copy.")


    # Move COCO json file to the proper 'annotations' folder, renaming it to remove "_coco" suffix
    final_json_target_name = f"instances_{split_name}2017.json"
    final_json_path = os.path.join(final_annotations_dir, final_json_target_name)
    shutil.move(coco_json_path_from_sahi, final_json_path)
    print(f"Moved COCO JSON to {final_json_path}")

    # Clean up: Move the initial SAHI output directory (which now only contains original images if not copied, or is empty) to an archive
    archive_base_path = os.path.join(current_sahi_dataset_path, "archive")
    os.makedirs(archive_base_path, exist_ok=True)
    
    # Destination for the archived initial output directory
    archived_initial_output_path = os.path.join(archive_base_path, os.path.basename(sahi_initial_output_dir))
    
    # If a previous run's archive exists for this specific folder, remove it to avoid error on move
    if os.path.exists(archived_initial_output_path):
        shutil.rmtree(archived_initial_output_path)
    shutil.move(sahi_initial_output_dir, archived_initial_output_path)
    print(f"Moved initial SAHI output directory {sahi_initial_output_dir} to {archived_initial_output_path}")

    return sliced_coco_data # This is the dict from the sliced JSON

def main(args):
    print(f"Using input COCO dataset from: {args.input_base_dir}")
    print(f"SAHI processed datasets will be saved under: {args.sahi_output_base_dir}")

    # Parse zoom factors string into a list of integers
    try:
        zoom_factors_list = [int(zf.strip()) for zf in args.zoom_factors.split(',')]
        if not zoom_factors_list or any(zf <= 0 for zf in zoom_factors_list):
            raise ValueError("Zoom factors must be positive integers.")
    except ValueError as e:
        print(f"Error: Invalid zoom_factors input: {args.zoom_factors}. {e}")
        return

    # Splits to process are fixed based on the original script's structure
    splits_for_processing = ["train", "val", "test"]

    # --- Sanity checks for the input dataset ---
    print("\nPerforming sanity checks on input dataset...")
    for split_name_check in splits_for_processing:
        annotation_file = os.path.join(args.input_base_dir, "annotations", f"instances_{split_name_check}2017.json")
        image_dir_check = os.path.join(args.input_base_dir, f"{split_name_check}2017")

        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found for sanity check: {annotation_file}. Skipping checks for this split.")
            continue
        if not os.path.exists(image_dir_check):
            print(f"Warning: Image directory not found for sanity check: {image_dir_check}. Skipping checks for this split.")
            continue
        
        print(f"Checking {split_name_check} data ({annotation_file})...")
        with open(annotation_file, "r") as file:
            data_to_check = json.load(file)
        check_multipolygons(data_to_check)
        check_negative_coords(data_to_check)
    print("Sanity checks completed.")

    # --- Calculate crop sizes based on zoom factors ---
    crop_zoom_sizes = [int(args.base_slice_size / zoom_factor) for zoom_factor in zoom_factors_list]

    # --- Main processing loop for SAHI slicing ---
    for crop_actual_size, zoom_factor_val in zip(crop_zoom_sizes, zoom_factors_list):
        print(f"\nProcessing for Zoom Factor: {zoom_factor_val} | Resulting Crop Size: {crop_actual_size}x{crop_actual_size}")
        
        for split_name_process in splits_for_processing:
            # Check if source data for this split exists before attempting to process
            input_annotation_file_check = os.path.join(args.input_base_dir, "annotations", f"instances_{split_name_process}2017.json")
            input_image_dir_check = os.path.join(args.input_base_dir, f"{split_name_process}2017")

            if not os.path.exists(input_annotation_file_check):
                print(f"  Warning: Input annotation file {input_annotation_file_check} not found. "
                      f"Skipping SAHI processing for '{split_name_process}' with crop size {crop_actual_size}.")
                continue
            if not os.path.exists(input_image_dir_check):
                print(f"  Warning: Input image directory {input_image_dir_check} not found. "
                      f"Skipping SAHI processing for '{split_name_process}' with crop size {crop_actual_size}.")
                continue
            
            print(f"  Applying SAHI slicing for split: {split_name_process}")
            
            # The returned sliced_coco_data is not explicitly used further by the original script's main loop,
            # but the function has side effects (creating the sliced dataset).
            _ = get_sahi_coco_data(
                input_base_dir_path=args.input_base_dir,
                crop_size_for_naming=crop_actual_size, # For naming the output folder, e.g., sahi_512_...
                crop_rows_val=crop_actual_size,
                crop_cols_val=crop_actual_size,
                split_name=split_name_process,
                sahi_output_base_path=args.sahi_output_base_dir,
                sahi_ignore_neg_samples_flag=args.sahi_ignore_negative_samples,
                overlap_ratio_val=args.overlap_ratio,
                min_area_ratio_val=args.min_area_ratio,
                sahi_verbose_flag=args.sahi_verbose
            )
    
    print("\nAll SAHI processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice a COCO dataset using SAHI for various zoom factors and organize outputs.")
    
    parser.add_argument('--input_base_dir', type=str, required=True, 
                        help='Path to the base directory of the input COCO dataset. '
                             'This directory should contain an "annotations/" subdirectory '
                             '(with instances_train2017.json, etc.) and image folders '
                             '(train2017/, val2017/, test2017/).')
    parser.add_argument('--sahi_output_base_dir', type=str, required=True,
                        help='Base directory where SAHI processed datasets will be saved. '
                             'Subdirectories like "sahi_{crop_size}_{neg_sample_handling}" will be created here.')
    parser.add_argument('--zoom_factors', type=str, default='2',
                        help='Comma-separated list of positive integer zoom factors (e.g., "1,2,4"). Default: "2".')
    parser.add_argument('--base_slice_size', type=int, default=1024,
                        help='Base size (pixels) used for calculating crop dimensions. '
                             'Actual crop size = base_slice_size / zoom_factor. Default: 1024.')
    parser.add_argument('--overlap_ratio', type=float, default=0.6,
                        help='Overlap ratio for slicing with SAHI (0.0 to 1.0). Default: 0.6.')
    parser.add_argument('--min_area_ratio', type=float, default=0.1,
                        help='Minimum area ratio for a sliced annotation to be kept by SAHI (0.0 to 1.0). Default: 0.1.')
    
    parser.add_argument('--sahi_ignore_negative_samples', action='store_true',
                        help='If set, SAHI will ignore negative samples (slices with no annotations). '
                             'If not set (default), negative samples are kept by SAHI, '
                             'and the output folder name will reflect this (e.g., "..._keep_neg").')
    parser.add_argument('--sahi_verbose', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable SAHI's verbose output during slicing. (Default: enabled). "
                             "Use --no-sahi-verbose to disable.")

    args = parser.parse_args()
    main(args)