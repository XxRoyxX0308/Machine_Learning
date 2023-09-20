from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import pygetwindow
import pyautogui
from PIL import Image,ImageEnhance
import numpy as np
import mouse
import os
from PIL import ImageGrab
import time
import cv2

model=YOLO("Pet2_best.pt")
#model.predict(source="0",show=True,conf=0.5)



while(1):
    image=pyautogui.screenshot(region=(0,0,1920,1080)) #左、上、寬、高
    image_cv=np.asarray(image) #PIL to numpy
    image_cv=cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB) 
    
    model.predict(source=image_cv,show=True,conf=0.5)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



##image_cv=cv2.imread("43766_01.jpg")
##model.predict(source=image_cv,show=True,conf=0.5)
##while(1):
##    a=0
