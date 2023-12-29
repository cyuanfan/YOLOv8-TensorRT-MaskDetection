import argparse
from pathlib import Path

import cv2
import numpy as np

from config import OBJECT_CLASSES, OBJECT_MASK_CLASSES, OBJECT_COLORS, OBJECT_MASK_COLORS
from models.utils import blob, det_postprocess, letterbox


def inference(engine, img, mask_detect_mode):
    H, W = engine.inp_info[0].shape[-2:]

    draw = img.copy()
    bgr, ratio, dwdh = letterbox(img, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = np.array(dwdh * 2, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor)
    # inference
    data = engine(tensor)

    bboxes, scores, labels = det_postprocess(data)
    bboxes -= dwdh
    bboxes /= ratio

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        if mask_detect_mode:
            cls = OBJECT_MASK_CLASSES[cls_id]
            color = OBJECT_MASK_COLORS[cls]
        else:
            cls = OBJECT_CLASSES[cls_id]
            color = OBJECT_COLORS[cls]  
        cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
        cv2.putText(draw,
                    f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, [225, 255, 255],
                    thickness=2)

    return draw



