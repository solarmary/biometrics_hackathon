import time
import cv2
import pandas as pd
import torch
import pickle
import yolov5.models

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class Predictor:
    def __init__(self):
        self.model= load("detector4/yolo5s6.pkl")

    def predict(self,img,arr_rez,idx):
        rez=0
        cnt=0
        for i in img:
            results = self.model(i)
            if [x for x in list(results.pandas().xyxy[0]['name']) if x in ['tv', 'cell phone', 'laptop', 'suitcase', 'book']]: # suitcase, microwave
                rez+=1
            cnt+=1
        rez=rez/cnt
        arr_rez[idx]=rez       
        return rez
