import json
from shapely import Polygon, is_valid
import numpy as np
import argparse

def main(args):
    with open(f"{args.dataset_path}/annotations/instances_train2017.json", "r") as f:
        data = json.load(f)

    invalids, valids, total = 0, 0, 0
    for m_annot in data['annotations']:
        polygons = []
        for polygon_idx in range(len(m_annot['segmentation'])):
            coco_coords = m_annot['segmentation'][polygon_idx]
            segment_coords = np.array(coco_coords).reshape(-1, 2)
            if len(segment_coords) >= 3:
                validity = is_valid(Polygon(segment_coords))
                if not validity:
                    invalids += 1
                else:
                    valids += 1
                total += 1

    print(f"Invalids: {invalids} ({invalids/total*100:.2f}%)")
    print(f"Valids: {valids} ({valids/total*100:.2f}%)")
    print(f"Total: {total}")

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