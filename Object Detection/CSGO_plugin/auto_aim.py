from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import pygetwindow
import pyautogui
import pydirectinput
from PIL import Image,ImageEnhance,ImageGrab
import numpy as np
import mouse
import keyboard
import ctypes
import win32api



model=YOLO("csgo4_best.pt")
#model.predict(source="0",show=True,conf=0.5)


def mousePosition(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        print (x,y)


switch=False
time_s=0
while(1):
    #image=pyautogui.screenshot(region=(560,290,800,500)) #左、上、寬、高
    image=ImageGrab.grab(bbox =(560,290,1360,790))
    #image=ImageGrab.grab(bbox =(0,0,1920,1080)) 
    image_cv=np.asarray(image) #PIL to numpy
    image_cv=cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB) 
    
    detect=model.predict(source=image_cv,show=False,conf=0.4)
    detect_np=detect[0].cpu().numpy().boxes.boxes
    #print(detect_np)
    
    #pyautogui.moveTo(100, 100, duration = 1.5) #用1.5秒移動到x=100，y=100的位置
    #cv2.namedWindow('mouse')
    #cv2.setMouseCallback('mouse',mousePosition)



    if keyboard.is_pressed("alt") and switch and time_s>100:
        switch=False
        time_s=0

    if keyboard.is_pressed("alt") and time_s>100:
        switch=True
        time_s=0

    time_s+=1

    print(switch)
    
    if switch and detect_np.size!=0:
        #print(detect_np[0][0])
        print("!!")
        
        mouse.get_position()
        position=pyautogui.position()
        
        #pyautogui.moveTo((detect_np[0][0]+detect_np[0][2])/2+560,detect_np[0][1]+290+10,duration=0.1)

        #mouse.move((detect_np[0][0]+detect_np[0][2])/2+560-position.x,detect_np[0][1]+290+10-position.y,absolute=False,duration=0.03)

        #ctypes.windll.user32.SetCursorPos(int((detect_np[0][0]+detect_np[0][2])/2+560),int(detect_np[0][1]+290+10))
        
        #win32api.SetCursorPos((int((detect_np[0][0]+detect_np[0][2])/2+560),int(detect_np[0][1]+290+10)))

        #pydirectinput.moveTo(int((detect_np[0][0]+detect_np[0][2])/2+560),int(detect_np[0][1]+290+10),duration=0.1)


        
        x=int(((detect_np[0][0]+detect_np[0][2])/2+560-960)/10)
        y=int((detect_np[0][1]+290-540+10)/10)
        if x==0:
            x=1
        if y==0:
            y=1

##        x=int(((detect_np[0][0]+detect_np[0][2])/2-960)/10)
##        y=int((detect_np[0][1]-540+10)/10)
##        if x==0:
##            x=1
##        if y==0:
##            y=1

        if keyboard.is_pressed("c"):
            pydirectinput.click(x,y,duration=0.1)
        else:
            pydirectinput.moveTo(x,y,duration=0.1)
        
    #if mouse.is_pressed("right"):
        #pydirectinput.moveTo(int((detect_np[0][0]+detect_np[0][2])/2),int(detect_np[0][1]+10),duration=0.5)
        #pydirectinput.moveTo(-10,-1,duration=1)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
