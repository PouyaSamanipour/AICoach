import torch
import trt_pose.models
import torch2trt
import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
outputPin=12
Start=22
pushbutton=40
GPIO.setup(outputPin,GPIO.OUT)
GPIO.setup(Start,GPIO.OUT)
GPIO.setup(pushbutton,GPIO.IN)
GPIO.output(outputPin,0)
GPIO.output(Start,0)
GPIO.output(outputPin,1)
import json
import trt_pose.coco
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))
WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
#######
#t0 = time.time()
#torch.cuda.current_stream().synchronize()
#for i in range(50):
#y = model_trtmo(data)
#torch.cuda.current_stream().synchronize()
#t1 = time.time()
#print(50.0 / (t1 - t0))

import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
###
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

###
#from jetcam.usb_camera import USBCamera
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
#camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)
camera.running = True
import ipywidgets
from IPython.display import display

image_w = ipywidgets.Image(format='jpeg')

display(image_w)

def execute(change):
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    return peaks
button=0

#Classifier
import os.path
from os import path
import pandas as pd
df_name = 'ai_coach_data_1.csv'
df_name_start = 'ai_coach_data_'
df_name_end = '.csv'
while path.exists(df_name):
  file_num = int(df_name[14:-4])
  new_file_num = file_num + 1
  df_name = df_name_start+str(new_file_num)+df_name_end
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.output(outputPin,1)
time.sleep(5)
GPIO.output(outputPin,0)
GPIO.output(Start,1)
time.sleep(5)
import numpy as np
counter=0
k=0
TH=0.17
Data=np.zeros((10000,37))
status=[]
display(image_w)
i=0
while (1==1):
    x=GPIO.input(pushbutton)
    if(x==0):
        GPIO.output(Start,0)
        time.sleep(0.5)
        GPIO.output(Start,1)
        peak=execute({'new': camera.value})
        for i in range (18):
            Data[counter][k:k+2]=peak[0][i][0]
            k=k+2

        if (Data[counter][8]==0 and Data[counter][12]==0) or (Data[counter][6]==0 and Data[counter][10]==0):
            print("Not detected")
            status.append("Not detected")
        elif (Data[counter][24]==0 and Data[counter][22]==0):
            print("Seated1")
            status.append("Seated1")
        elif (Data[counter][24]!=0 and Data[counter][22]==0):
            if Data[counter][28]!=0:
                if (Data[counter][28]-Data[counter][24])>=TH:
                    print("Standing position1")
                    status.append("Standing position1")
                else:
                    if ((Data[counter][28]-Data[counter][24])>0.65*(Data[counter][24]-Data[counter][12])):
                        print("Standing position2")
                        status.append("Standing position2")
                    else:
                        print("seated2")
                        status.append("seated2")
                    
            else:
                    print("seated3")
                    status.append("seated3")
        elif (Data[counter][24]==0 and Data[counter][22]!=0):
            if Data[counter][26]!=0:
                if (Data[counter][26]-Data[counter][22])>=TH:
                    print("Standing position3")
                    status.append("Standing position3")
                else:
                    if((Data[counter][26]-Data[counter][22])>0.65*(Data[counter][22]-Data[counter][10])):
                        print("Standing position4")
                        status.append("Standing position4")
                    else:
                        print("seated4")
                        status.append("seated4")
   
            else:
                print("seated5") 
                status.append("seated5")
        elif (Data[counter][24]!=0 and Data[counter][22]!=0):
            if Data[counter][28]!=0:
                if (Data[counter][28]-Data[counter][24]>=TH) or (Data[counter][26]-Data[counter][22]>=TH):
                    print("Standing position5")
                    status.append("Standing position5")
                else:
                    if ((Data[counter][26]-Data[counter][22])>0.65*(Data[counter][22]-Data[counter][10])) or ((Data[counter][28]-Data[counter][24])>0.65*(Data[counter][24]-Data[counter][12])):
                        print("Standing position6")
                        status.append("Standing position6")
                    else:    
                        print("seated6")
                        status.append("seated6")
            else:
                print("seated7")
                status.append("seated7")
        counter=counter+1
        k=0
        time.sleep(3)
        camera.unobserve_all()
    elif(x==1):
        GPIO.output(Start,0)
        camera.unobserve_all()
i=i+1        
