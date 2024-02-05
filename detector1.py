import cv2
import numpy as np
from facenet_pytorch import MTCNN
import math

mtcnn = MTCNN(keep_all=True)
face_cascade = cv2.CascadeClassifier('detector1/haarcascade_frontalface_alt.xml')

#https://www.tutorialspoint.com/how-to-detect-a-face-and-draw-a-bounding-box-around-it-using-opencv-python
def face_detect(img):
    if img is None: return None
    rez=mtcnn.detect(img,landmarks=False)
    if rez is None or rez[0] is None: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1,2,minSize=(30,30))
        if faces is None or len (faces)==0: return None
        rez = [[(x,y,x+w,y+h) for x,y,w,h in faces ]]
    x0_,y0_,x1_,y1_=0,0,0,0
    for face in rez[0]:
        x0,y0,x1,y1 = int(face[0]),int(face[1]),int(face[2]),int(face[3])
        if(x1_ - x0_ <  x1 - x0 ): 
            x0_,y0_,x1_,y1_=x0,y0,x1,y1
    return x0_,y0_,x1_-x0_,y1_-y0_

# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
def find_boxes(img,face_boxes,name=None):
    if face_boxes is None or img is None: return 0.5
    if not name is None:
        dir="test/"+name.split('.')[0]
        os.makedirs(dir,exist_ok=True)
    height, width, channels = img.shape
    result = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,71,7)
    
    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)
    
    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    
    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    detect_box_in_box=0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if not face_boxes is None:
            if x-x//7<=face_boxes[0] and y-y//7<=face_boxes[1] and w+w//10>=face_boxes[2] and h+h//10>=face_boxes[3]:   # !!!
                detect_box_in_box+=1
        if detect_box_in_box==0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (12,12,255), 3)
    if not name is None:        
        # cv2.imwrite(dir+'/thresh.jpg', thresh)
        # cv2.imwrite(dir+'/opening.jpg', opening)
        cv2.imwrite(dir+'/boxes.jpg', img)
    return 0 if detect_box_in_box == 0 else 1

class Predictor:
    def __init__(self):
        pass
        
    def predict(self,images,arr_rez,idx):
        rez=0
        cnt=0
        for img in images:
            facebox=face_detect(img)
            rez+=find_boxes(img,facebox)
            cnt+=1
        rez=rez/cnt
        arr_rez[idx]=rez
        return rez
