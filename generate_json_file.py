import json
import cv2

img_path='p1.jpg'
frame_orig = cv2.imread(img_path)
data = {'img':frame_orig.tolist()}

with open("input.json","w") as f:
   json.dump(data,f)
