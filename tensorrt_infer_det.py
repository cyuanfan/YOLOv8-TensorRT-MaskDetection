from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import OBJECT_CLASSES, OBJECT_MASK_CLASSES, OBJECT_COLORS, OBJECT_MASK_COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox


def inference(engine, img, mask_detect_mode, cuda_device):
    device = torch.device(cuda_device)
    Engine = TRTModule(engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    draw = img.copy()
    bgr, ratio, dwdh = letterbox(img, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    # inference
    data = Engine(tensor)

    bboxes, scores, labels = det_postprocess(data)
    bboxes -= dwdh
    bboxes /= ratio

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().int().tolist()
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
