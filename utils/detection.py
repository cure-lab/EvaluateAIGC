import sys
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image
import numpy as np

class Detection():
    def __init__(self, target):
        self.model_dict = {"face":"face_yolov8m.pt", "hand":"hand_yolov8s.pt", "body":"person_yolov8m-seg.pt"}
        self.set_model(target)

    def set_model(self, target):
        assert target in self.model_dict.keys(), "target should be chosen from [face, hand, body]"
        path = hf_hub_download("Bingsu/adetailer", self.model_dict[target])
        self.model = YOLO(path)
    
    def detect(self, path):
        img_pil = Image.open(path)
        output = self.model(img_pil)
        bboxs = [[int(i) for i in bbox] for bbox in output[0].boxes.xyxy.cpu().numpy()]
        return bboxs

    def crop_image(self, path, padding=0):
        bboxs = self.detect(path)
        imgs = []
        for bbox in bboxs:
            left = max(bbox[0]-padding, 0)
            top = max(bbox[1]-padding, 0) 
            right = min(bbox[2]+padding, img.shape[1])
            bottom = min(bbox[3]+padding, img.shape[0])
            img = Image.fromarray(np.array(img)[top:bottom,left:right])
            imgs.append([bbox, img])
        return imgs