import cv2
import time
import os
import sys
import threading
from threading import Thread
from time import sleep
import numpy as np
import detector1
import detector2
import detector3
#import detector4
import detector5
from detector3 import ImgNet


d1 = detector1.Predictor() # предиктор рамок вокруг лица, на чистом CV
d2 = detector2.Predictor() # предиктор отслеживания изменения фона на фото и окружающе сцены, модель RandomForest
d3 = detector3.Predictor() # предиктор с механизмом внимания
#d4 = detector4.Predictor() # предиктор телефона(планшета, ноута) в кадре, на облегченной Yolo
d5 = detector5.Predictor() # предиктор цветового балланса в кадре, на DecisionTree

# служебные функции
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    if image is None: return None
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
def avg(arr):
    arr=[x for x in arr if not x is None]
    return np.average(arr)

# главная функция. ее нужно запускать для теста. Аргумент путь до односекундного видео.
# Возвращает вероятность атаки предъявлением (0 - точно нет атаки, 1 - атака точно есть) и время обработки видео
def main(one_sec_video):
    cap = cv2.VideoCapture(one_sec_video)
    rez=np.array([None,None,None,None,None,None,])
    threads=[]
    images=[]
    images_=[]
    cnt=0
    while True:
        ret, img_ = cap.read()
        cnt+=1
        if cnt % 8!=0: continue
        img=image_resize(img_,width=450)
        if not ret or len(images)>=3: break
        images.append(img)
        images_.append(img_)
    start=time.time()
    threads.append(Thread(target=d1.predict, args=(images, rez, 1,))) #  можно закомментирвать две строки, чтобы отключить предиктор
    threads[-1].start()
    threads.append(Thread(target=d2.predict, args=(images, rez, 2,)))  #  можно закомментирвать две строки, чтобы отключить предиктор
    threads[-1].start()
    threads.append(Thread(target=d3.predict, args=(images_, rez, 3,)))  #  можно закомментирвать две строки, чтобы отключить предиктор
    threads[-1].start()
    #threads.append(Thread(target=d4.predict, args=(images_, rez, 4,)))  #  можно закомментирвать две строки, чтобы отключить предиктор
    #threads[-1].start()
    threads.append(Thread(target=d5.predict, args=(images_, rez, 5,)))  #  можно закомментирвать две строки, чтобы отключить предиктор
    threads[-1].start()
    for i in range(100):
       cnt=0
       for t in threads:
           cnt+= 1 if not t is None and t.is_alive() else 0
       if cnt==0: break
       sleep(0.01)
    rez_= avg(rez)
    stop=time.time()
    text = "атака обнаружена" if rez_ >= 0.4 else "атака не обнаружена"
    print(f"{text}     Вероятность атаки: {rez_}      Время на анализ (сек): {stop-start}s")
    return rez_,stop-start

if(len(sys.argv)!=2):
    print("укажите односекундный видеофайл для анализа")
    sys.exit(-1)
main(sys.argv[-1])
