import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg, K=100):
    batch, cat, _, _ = heat.size()
    
    # perform nms on heatmaps
    heat = _nms(heat) 
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections


class COCOEvaluator(object):
    def __init__(self, annoth_path, cat_ids, filename="inference.json"):

        self.ids_cats = {v: k for k, v in cat_ids.items()} 
        self.annot_path = annoth_path 
        self.jdict = []
        self.filename = Path(filename)

    def add_img_dets(self, image_id, detections):
        """ 
        [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
        """
        for det in detections:
            w, h = float(det[2] - det[0]), float(det[3] - det[1])
            if w > 0 and h > 0:
                self.jdict.append({'image_id': int(image_id),
                                'category_id': self.ids_cats[int(det[5])],
                                'bbox': [float(det[0]), float(det[1]), w, h],
                                'score': float(det[4])})

    def evaluate(self):
        print(f"\nEvaluating pycocotools mAP.")
        if not self.jdict:
            return 0.0, 0.0
        
        print(f"\nSaving {self.filename}...")
        with open(self.filename, 'w') as f:
            json.dump(self.jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(self.annot_path)  # init annotations api
            pred = anno.loadRes(str(self.filename))  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            return map, map50
        except Exception as e:
            print(f'pycocotools unable to run: {e}')