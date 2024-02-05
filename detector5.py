import pickle
import cv2
import PIL.Image as Image
from PIL import ImageOps
import numpy as np
from facenet_pytorch import MTCNN
import math

WIDTH=200
mtcnn = MTCNN(keep_all=True)
face_cascade = cv2.CascadeClassifier('detector5/haarcascade_frontalface_alt.xml')

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

def vectorize(img):
    if img is None: return None
    t_16=2**16
    t_8=2**8
    arr=np.asarray(img).copy()
    arr[:,:,0]*=t_16
    arr[:,:,1]*=t_8
    h,w,_=arr.shape
    rez1=np.zeros(h*w)
    # rez2=np.zeros(h*w)
    cnt=0
    arr=arr.reshape(-1)
    for i in range(0,len(arr),3):
        color1=arr[i]+arr[i+1]+arr[i+2]
        rez1[cnt]=color1
        # color2=(arr[i]*t_16)+(arr[i+1]*t_8)+arr[i+2]
        # rez2[cnt]=color2
        cnt+=1
    return rez1#,rez2        
    
class Predictor:
    def __init__(self):
        self.model = load("detector5/dt.pkl")

    def predict(self,img,arr_rez,idx):
        rez=0
        cnt=0
        for i in img:
            i=resize_crop(i)
            vec=vectorize(i)
            if vec is None: 
                print("face not found!!!")
                continue
            rez+=self.model.predict([vec])
            cnt+=1
        rez=rez[0]    
        arr_rez[idx] = rez/cnt if cnt!=0 else None
        return rez/cnt if cnt!=0 else None
