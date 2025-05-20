# Current outputs -> need to correct the invalid polygons or our results won't be very good
import json
from shapely import Polygon, is_valid
import numpy as np

DATASET_JSON = "/h/jquinto/LIFEPLAN_SAHI_tiling/sahi_datasets_SR_swinir_bioscan/sahi_1024_ignore_neg_SR"

with open(f"{DATASET_JSON}/annotations/instances_train2017.json", "r") as f:
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