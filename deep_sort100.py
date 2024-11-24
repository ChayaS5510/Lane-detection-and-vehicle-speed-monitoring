import os
import torch
import numpy as np
import cv2
from deep_sort.deep_sort import DeepSort

def initialize_deep_sort():
    model_path = 'path_to_deep_sort_model'
    deepsort = DeepSort(model_path)
    return deepsort

def run_deep_sort(encoder, tracker, frame, detections):
    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, conf, cls in detections:
        obj = [x1, y1, x2-x1, y2-y1]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls)
    
    bbox_xywh = torch.Tensor(bbox_xywh)
    confs = torch.Tensor(confs)
    clss = torch.Tensor(clss)
    
    outputs = tracker.update(bbox_xywh, confs, clss, frame)
    
    return tracker
