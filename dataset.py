from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
import os
import math
import random
from pycocotools import coco


def rad(a, b, c):
    sq = np.sqrt(b**2 - 4 * a * c)
    return (b + sq) / 2


def gaussian_radius(det_size, min_overlap=0.7):
    width, height = det_size
    
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = rad(a1, b1, c1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    r2 = rad(a2, b2, c2)
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    r3 = rad(a3, b3, c3)

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    
    # zeroing the negligible values
    eps = np.finfo(h.dtype).eps * h.max()
    h[h < eps] = 0
    
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    # find the portion of the heatmap to past the gaussian to
    x, y = center
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    # paste the gaussian onto the heatmap
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


class CTDetDataset(torch.utils.data.Dataset):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3) 
    
    def __init__(self, cfg, data_root, split, debug=False):
        super(CTDetDataset, self).__init__()
        self.cfg = cfg
        self.data_dir = data_root
        self.split = split
        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_{}2017.json').format(split)

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.img_dir = os.path.join(self.data_dir, 'images', '{}2017').format(split)
        
        self.max_objs = 128
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
        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    def __len__(self):
        return self.num_samples
  
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def __getitem__(self, index):
        
        # read the image and the annotations
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)

        input_w, input_h, = self.cfg['input_w'], self.cfg['input_h'] 
        inp = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR) 

        # normalize and to_tensor 
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        
        output_w, output_h = input_w // self.cfg['down_ratio'], input_h // self.cfg['down_ratio']
        out_scale = np.array([output_w / img.shape[1], output_h / img.shape[0]])

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])

            # resize the box to the input size
            bbox[[0, 2]] = bbox[[0, 2]] * out_scale[0]
            bbox[[1, 3]] = bbox[[1, 3]] * out_scale[1]
            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h <= 0 and w <= 0:
                continue
            
            radius = gaussian_radius((math.ceil(w), math.ceil(h)))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            
            draw_umich_gaussian(hm[cls_id], ct_int, int(radius))
            wh[k] =  w, h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}
        meta = {'img_id': img_id, 'img_sz': np.array(img.shape[:2]), 'out_scale': out_scale}
        ret.update({'meta' : meta})

        return ret