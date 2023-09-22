#import os
#from IPython import display
#display.clear_output()
#from ultralytics import YOLO
#from IPython.display import display,Image
#!yolo mode=checks
#!yolo task=segment mode=predict model=yolov8l-seg.pt conf=0.4 source='https://media.roboflow.com/notebooks/examples/dog.jpeg'

from roboflow import Roboflow
rf = Roboflow(api_key="6lrTYPxR5CWTZyGLC94a")
project = rf.workspace("csgo-aimbot-fe6g4").project("csgo-models-umdcn")
dataset = project.version(3).download("yolov8")
