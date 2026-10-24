import json
from shapely import Polygon, is_valid
import numpy as np
import argparse
import sys

SPLITS = ["train", "val", "test"]

def count_polygons(annotations):
    invalids, valids = 0, 0
    for m_annot in annotations:
        for polygon_idx in range(len(m_annot['segmentation'])):
            coco_coords = m_annot['segmentation'][polygon_idx]
            segment_coords = np.array(coco_coords).reshape(-1, 2)
            if len(segment_coords) >= 3:
                if is_valid(Polygon(segment_coords)):
                    valids += 1
                else:
                    invalids += 1
    return invalids, valids

def main(args):
    any_invalid = False
    for split in SPLITS:
        with open(f"{args.dataset_path}/annotations/instances_{split}2017.json", "r") as f:
            data = json.load(f)

        invalids, valids = count_polygons(data['annotations'])
        total = invalids + valids
        any_invalid = any_invalid or invalids > 0

        print(f"[{split}]")
        if total == 0:
            print("No polygons")
            continue
        print(f"Invalids: {invalids} ({invalids/total*100:.2f}%)")
        print(f"Valids: {valids} ({valids/total*100:.2f}%)")
        print(f"Total: {total}")

    if any_invalid:
        print("Invalid polygons remain; see the counts above.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check post-processed data for any remaining invalid annotations')
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path of postprocessed data'
    )
    args = parser.parse_args()
    main(args)
