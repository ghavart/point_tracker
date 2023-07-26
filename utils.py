import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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


def ctdet_decode(heat, wh, reg, cat_spec_wh=False, K=100):
    batch, cat, _, _ = heat.size()
    
    # perform nms on heatmaps
    heat = _nms(heat) 
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    # reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    
    # wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections


# def affine_transform(pt, t):
#     new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
#     new_pt = np.dot(t, new_pt)
#     return new_pt[:2]


# def get_3rd_point(a, b):
#     direct = a - b
#     return b + np.array([-direct[1], direct[0]], dtype=np.float32)


# def get_dir(src_point, rot_rad):
#     sn, cs = np.sin(rot_rad), np.cos(rot_rad)

#     src_result = [0, 0]
#     src_result[0] = src_point[0] * cs - src_point[1] * sn
#     src_result[1] = src_point[0] * sn + src_point[1] * cs

#     return src_result


# def get_affine_transform(center,
#                          scale,
#                          rot,
#                          output_size,
#                          shift=np.array([0, 0], dtype=np.float32),
#                          inv=0):
#     if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
#         scale = np.array([scale, scale], dtype=np.float32)

#     scale_tmp = scale
#     src_w = scale_tmp[0]
#     dst_w = output_size[0]
#     dst_h = output_size[1]

#     rot_rad = np.pi * rot / 180
#     src_dir = get_dir([0, src_w * -0.5], rot_rad)
#     dst_dir = np.array([0, dst_w * -0.5], np.float32)

#     src = np.zeros((3, 2), dtype=np.float32)
#     dst = np.zeros((3, 2), dtype=np.float32)
#     src[0, :] = center + scale_tmp * shift
#     src[1, :] = center + src_dir + scale_tmp * shift
#     dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
#     dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

#     src[2:, :] = get_3rd_point(src[0, :], src[1, :])
#     dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

#     if inv:
#         trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
#     else:
#         trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

#     return trans


# def transform_preds(coords, center, scale, output_size):
#     target_coords = np.zeros(coords.shape)
#     trans = get_affine_transform(center, scale, 0, output_size, inv=1)
#     for p in range(coords.shape[0]):
#         target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
#     return target_coords


# def ctdet_post_process(dets, c, s, h, w, num_classes):
#   # dets: batch x max_dets x dim
#   # return 1-based class det dict
#   ret = []
#   for i in range(dets.shape[0]):
#     top_preds = {}
#     dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w[i], h[i]))
#     dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w[i], h[i]))
#     classes = dets[i, :, -1]
#     for j in range(num_classes):
#       inds = (classes == j)
#       top_preds[j + 1] = np.concatenate([
#         dets[i, inds, :4].astype(np.float32),
#         dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
#     ret.append(top_preds)
#   return ret



class COCOEvaluator(object):
    def __init__(self, data_root, split='test', filename="inference.json"):
        
        
        self.data_dir = data_root
        self.split = split
        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json').format(split)
        elif split == 'val':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_{}2017.json').format(split)
        else:
            raise NotImplementedError 
        
        self.jdict = []
        self.filename = Path(filename)

        self.class_name = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.num_classes = len(self.class_name)

    def add_img_dets(self, image_id, detections):
        """ 
        [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
        """
        for det in detections:
            w = det[2] - det[0]
            h = det[3] - det[1]
            if w > 1 and h > 1:
                self.jdict.append({'image_id': int(image_id),
                                'category_id': self.class_name[int(det[5])],
                                'bbox': [int(det[0]), int(det[1]), int(w), int(h)],
                                'score': str(round(det[4], 5))})

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



    # def xywh2center_net(xywh: np.ndarray, out_w: int, out_h: int, num_classes: int) -> tuple(np.ndarray, np.ndarray):
    #     """
    #     Convert xywh to:
    #         1. Centroids 2d gausians: (num_classes, out_h, out_w)
    #         2. Width and height: (2, out_h, out_w)
    #         3. Offsets: (2, out_h, out_w)
    #     """
    #     x, y, w, h = xywh




    # def center_net2xywh(centroids: np.ndarray, wh: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    #     """
    #     Convert:
    #         1. Centroids 2d gausians: (num_classes, out_h, out_w)
    #         2. Width and height: (2, out_h, out_w)
    #         3. Offsets: (2, out_h, out_w)
    #     to xywh
    #     """
    #     pass