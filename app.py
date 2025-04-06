from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO('FRUIT_MODEL.pt')

result = model(source=0, show = True, conf= 0.2, save = False, imgsz=1024) #reduce imagesz if it lags