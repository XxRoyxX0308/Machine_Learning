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
import win32con
import time



model=YOLO("AimLab_best.pt")
#model.predict(source="0",show=True,conf=0.5)


check=False
time_c=100
switch=False
time_s=100
while(1):
    #image=pyautogui.screenshot(region=(560,290,800,500)) #左、上、寬、高
    
    #image=ImageGrab.grab(bbox =(560,290,1360,790))
    image=ImageGrab.grab(bbox =(0,0,1920,1080)) 
    image_cv=np.asarray(image) #PIL to numpy
    image_cv=cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB) 
    
    detect=model.predict(source=image_cv,show=False,conf=0.4)
    detect_np=detect[0].cpu().numpy().boxes.boxes
    #print(detect_np)
    
    #pyautogui.moveTo(100, 100, duration = 1.5) #用1.5秒移動到x=100，y=100的位置
    #cv2.namedWindow('mouse')
    #cv2.setMouseCallback('mouse',mousePosition)

##    if keyboard.is_pressed("c"):
##        ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN)
##        ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP)
##        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 10, 10, 0, 0)



    if keyboard.is_pressed("alt") and switch and time_s>100:
        switch=False
        time_s=0

    if keyboard.is_pressed("alt") and time_s>100:
        switch=True
        time_s=0
        
    time_s+=1
    
    if keyboard.is_pressed("c") and check==False and time_c>100:
        check=True
        time_c=0

    if keyboard.is_pressed("c") and check==True and time_c>100:
        check=False
        time_c=0

    time_c+=1

    #print(switch)
    
    if switch and detect_np.size!=0:
        position=pyautogui.position()

        x=int(((detect_np[0][0]+detect_np[0][2])/2)-position[0])
        y=int(((detect_np[0][1]+detect_np[0][3])/2)-position[1])

##        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 2, 2)
##        ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN)
##        ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP)
        

        if check:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
            ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN)
            ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP)
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
