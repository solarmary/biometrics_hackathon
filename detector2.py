import pickle
from facenet_pytorch import MTCNN
import cv2
import math
import numpy as np
from PIL import Image

def resize(frame,w,h):
  return Image.fromarray(frame).resize((w, h))

def meanPixels(box):
  newBox = np.array([0.]*box.shape[0]*box.shape[1])
  s = 0
  for lineI in range(len(box)):
    for pixelI in range(len(box[lineI])):
      pixel = box[lineI][pixelI]
      newBox[s] = (pixel[0]<<16) + (pixel[1]<<8) + pixel[2]
      s += 1
  return newBox

def crop_image(image, coordinates):
    cropped_image = Image.fromarray(image).crop(coordinates)
    return cropped_image

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class Predictor:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.new_w = 50
        self.new_h = 50
        self.size = self.new_w * self.new_h
        self.modelTop = load("detector2/top.pkl")
        self.modelBot = load("detector2/bot.pkl")
        self.modelLeft = load("detector2/left.pkl")
        self.modelRight = load("detector2/right.pkl")

    def predict(self,images,arr_rez,idx):
        count = 0.
        answersSum = 0.
        for img in images:
          try:
            frame = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            rez=self.mtcnn.detect(frame,landmarks=True)
            n=0
            x0 = math.floor(rez[0][n][0])
            y0 = math.floor(rez[0][n][1])
            x1 = math.floor(rez[0][n][2])
            y1 = math.floor(rez[0][n][3])

            h=img.shape[0]
            w=img.shape[1]

            coordinatesBot = (0, y1, w, h)
            coordinatesTop = (0, 0, w, y0)
            coordinatesLeft = (0, y0, x0, w)
            coordinatesRight = (x1, y0, w, w)

            bot = crop_image(img,coordinatesBot)
            top = crop_image(img,coordinatesTop)
            left = crop_image(img,coordinatesLeft)
            right = crop_image(img,coordinatesRight)

            topMeanPixels = meanPixels(np.asarray(resize(np.asarray(top),self.new_w,self.new_h)))
            botMeanPixels = meanPixels(np.asarray(resize(np.asarray(bot),self.new_w,self.new_h)))
            leftMeanPixels = meanPixels(np.asarray(resize(np.asarray(left),self.new_w,self.new_h)))
            rightMeanPixels = meanPixels(np.asarray(resize(np.asarray(right),self.new_w,self.new_h)))
            answers = [self.modelTop.predict([topMeanPixels]),self.modelBot.predict([botMeanPixels]),self.modelRight.predict([rightMeanPixels]),self.modelLeft.predict([leftMeanPixels])]
            r = answers.count(1)
            noR = answers.count(0)
            if r >= noR:
              answersSum += 1
            count += 1
          except:
            continue
        arr_rez[idx]=  answersSum/count    
        return answersSum/count