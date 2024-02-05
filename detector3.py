import pickle
import cv2
import PIL.Image as Image
from PIL import ImageOps
import numpy as np
from facenet_pytorch import MTCNN
import math
import torch
from torch.utils.data import Dataset
import torch
from torch.nn import Module,Linear,Conv2d,Sequential,ReLU,ELU,Dropout,BatchNorm1d,BatchNorm2d,Flatten,CrossEntropyLoss,AdaptiveAvgPool1d,MaxPool1d,MaxPool2d
from torch.optim import lr_scheduler,SGD,Adam,AdamW,Adagrad,RMSprop
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

WIDTH=200
mtcnn = MTCNN(keep_all=True)
face_cascade = cv2.CascadeClassifier('detector3/haarcascade_frontalface_alt.xml')


class ImgNet(Module):
    
    def __init__(self, num_classes):   
        super(self.__class__,self).__init__()
        self.num_classes=num_classes
        self.BIND_SIZE=441
        self.l1=Sequential(Conv2d(3,20,5),MaxPool2d(3),BatchNorm2d(20),ReLU(), Dropout(0.1))
        self.l2=Sequential(Conv2d(20,40,3),MaxPool2d(3),BatchNorm2d(40),ReLU(), Dropout(0.1))
        # self.l3=Sequential(Conv2d(40,60,3),MaxPool2d(3),BatchNorm2d(60),ReLU(), Dropout(0.2))
        self.f=torch.nn.Flatten(start_dim=2)
        self.c= torch.nn.Conv1d(self.BIND_SIZE, 1, 1)
        self.sm = torch.nn.Softmax(2)
        self.l=Linear(self.BIND_SIZE,self.num_classes)
        self.b=BatchNorm1d(self.num_classes)

    
    def forward(self, x):
        x=self.l2(self.l1(x))
        # extractr features vectors
        features=self.f(x)
        features=features.permute(0,2,1)
        # print(x1.shape)
        # calculate attention
        att_score=self.c(features)
        # print(x2.shape)  
        att_score=self.sm(att_score)
        # print(x2.shape)
        # aggregate
        features=features.permute(0,2,1)
        x=torch.bmm(att_score,features)
        # predict 
        # print(x.shape)
        x=torch.squeeze(x,1)
        # print(x.shape)
        x=self.l(x)
        x=self.b(x)
        return x


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
    
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
    
def resize_crop(img):
    if img is None: return None
    if img.shape[1]<img.shape[0]:
        img = image_resize(img,width = int(WIDTH))  # сжимаем!
    else: 
        img = image_resize(img,height = int(WIDTH))  # сжимаем!
    f=face_detect(img)
    if f is None: 
        print("Warning: face not found !   Use center image, as face.")
        w_,h_=img.shape[1],img.shape[0]
        cx0_,cy0_=w_//2,h_//2
        f=[cx0_-w_//4,cy0_-h_//4,w_//2,h_//2]
    # print(f)
    w=f[2]
    h=f[3]
    if w<WIDTH/15: 
        print("Warning: face too  small!       Use center image, as face.")
        w_,h_=img.shape[1],img.shape[0]
        cx0_,cy0_=w_//2,h_//2
        f=[cx0_-w_//4,cy0_-h_//4,w_//2,h_//2]
    cx0=f[0]+w//2
    cy0=f[1]+h//2
    # print(cx0,cy0)
    mat = np.float32([ [1,0,img.shape[1]//2-cx0], [0,1,img.shape[0]//2-cy0] ])   
    img = cv2.warpAffine(img, mat, dsize=(img.shape[1],img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    img= img[img.shape[0]//2-WIDTH//2:img.shape[0]//2+WIDTH//2, img.shape[1]//2-WIDTH//2:img.shape[1]//2+WIDTH//2]
    return img

    
    
class Predictor:
    def __init__(self):
        self.model = torch.load("detector3/att.pt")
        self.model.eval()
        self.transform = transforms.ToTensor()
        self.sm=torch.nn.Softmax(dim=1)

    def predict(self,img,arr_rez,idx):
        rez=0
        cnt=0
        for i in img:
            vec=resize_crop(i)
            if vec is None: 
                print("face not found!!!")
                continue
            vec=self.transform(vec).float()
            r=self.model(vec[None, :])
            rez+=self.sm(r)[0,1].item()
            cnt+=1
        arr_rez[idx] = rez/cnt if cnt!=0 else None
        return rez/cnt if cnt!=0 else None
