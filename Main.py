import torch
import trt_pose.models
import torch2trt
import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
outputPin=12
Start=22
pushbutton=13
Recording=36
GPIO.setup(outputPin,GPIO.OUT)
GPIO.setup(Start,GPIO.OUT)
GPIO.setup(Recording,GPIO.OUT)
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
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
#######
#t0 = time.time()o
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
    return peaks, image
button=0
#creating csv file
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
#Classifier
import RPi.GPIO as GPIO
import time
from datetime import datetime
GPIO.setmode(GPIO.BOARD)
time.sleep(1)
#GPIO.output(outputPin,0)
GPIO.output(Start,1)
time.sleep(1)
import numpy as np
counter=0
k=0
TH=0.17
Data=np.zeros((1000000,37))
status=[]
timestamp=[]
Identifier=[]
#display(image_w)
i=0
l=0
summation=0
state_old="Not detected"
while (1==1):
    time.sleep(0.1)
    x=GPIO.input(pushbutton)
    if(x==0):
        GPIO.output(Recording,1)
        peak=execute({'new': camera.value})
        for i in range (18):
            Data[counter][k:k+2]=peak[0][0][i][0]
            k=k+2
        if (counter<10):
                GPIO.output(Recording,0)
                time.sleep(0.1)
                GPIO.output(Recording,1)
                if Data[counter][12]!=0:
                    summation=summation+Data[counter][12]
                    l=l+1
                    Shoulder_mean=summation/l
                status.append("Seated Calibration")
                state_new="Seated Calibration" 
                #print(Shoulder_mean)
        else:
            GPIO.output(Recording,1)  
            if (Data[counter][8]==0 and Data[counter][12]==0) or (Data[counter][6]==0 and Data[counter][10]==0):
                print("Not detected")
                status.append("Not detected")
                state_new="Not detected"
            elif (Data[counter][24]==0 and Data[counter][22]==0):
                if (Data[counter][10]<=0.15 or Data[counter][12]<=0.15):
                    status.append("Standing position new")
                    print("Standing position new")
                    state_new="Standing position"
                else:
                    status.append("Seated1")
                    print("Seated")
                    state_new="Seated"
            elif (Data[counter][24]!=0 and Data[counter][22]==0):
                if Data[counter][28]!=0:
                    if (Data[counter][28]-Data[counter][24])>=TH:
                        print("Standing position1")
                        status.append("Standing position1")
                        state_new="Standing position"
                    else:
                        if ((Data[counter][28]-Data[counter][24])>0.65*(Data[counter][24]-Data[counter][12])):
                            print("Standing position2")
                            status.append("Standing position2")
                            state_new="Standing position"
                        else:
                            print("seated2")
                            status.append("seated2")
                            state_new="Seated"
                else:
                    if (Data[counter][10]<=0.20 or Data[counter][12]<=0.2):
                        status.append("Standing position new_1")
                        print("Standing position new_1")
                        state_new="Standing position"
                    else:
                        print("seated3")
                        status.append("seated3")
                        state_new="Seated"
            elif (Data[counter][24]==0 and Data[counter][22]!=0):
                if Data[counter][26]!=0:
                    if (Data[counter][26]-Data[counter][22])>=TH:
                        print("Standing position3")
                        status.append("Standing position3")
                        state_new="Standing position"
                    else:
                        if((Data[counter][26]-Data[counter][22])>0.65*(Data[counter][22]-Data[counter][10])):
                            print("Standing position4")
                            status.append("Standing position4")
                            state_new="Standing position"
                        else:
                            print("seated4")
                            status.append("seated4")
                            state_new="Seated"
    
                else:
                        if (Data[counter][10]<=0.20 or Data[counter][12]<=0.2):
                            status.append("Standing position new_2")
                            print("Standing position new_2")
                            state_new="Standing position"
                        else:
                            print("seated5") 
                            status.append("seated5")
                            state_new="Seated"
            elif (Data[counter][24]!=0 and Data[counter][22]!=0):
                if Data[counter][28]!=0:
                    if (Data[counter][28]-Data[counter][24]>=TH) or (Data[counter][26]-Data[counter][22]>=TH):
                        print("Standing position5")
                        status.append("Standing position5")
                        state_new="Standing position"
                    else:
                        if ((Data[counter][26]-Data[counter][22])>0.65*(Data[counter][22]-Data[counter][10])) or ((Data[counter][28]-Data[counter][24])>0.65*(Data[counter][24]-Data[counter][12])):
                            print("Standing position6")
                            status.append("Standing position6")
                            state_new="Standing position"
                        else:    
                            print("seated6")
                            status.append("seated6")
                            state_new="Seated"
                else:
                    if (Data[counter][10]<=0.20 or Data[counter][12]<=0.2):
                        status.append("Standing position new_3")
                        print("Standing position new_3")
                        state_new="Standing position"
                    else:
                        print("seated7")
                        status.append("seated7")
                        state_new="Seated"
        counter=counter+1
        from random import randint
        dummy=randint(0,9999999999)
        current_time=datetime.now()
        timestamp.append(current_time)
        Identifier.append(dummy)
        d={'col1':status,'col2':timestamp,'col3':Identifier}
        df = pd.DataFrame(data=d)
        df.to_csv(df_name)
        k=0
        time.sleep(0.1)
        # saving images
        if (counter%900==1 or state_new!=state_old):
            Image_name=current_time.strftime("%m_%d_%Y,%H:%M:%S")
            #Image_sufix=".png"
            #Image_name+=Image_sufix
            from PIL import Image as im
            data=im.fromarray(peak[1])
            data.save('Image_'+Image_name+'_'+str(dummy)+'.png')
        state_old=state_new
        camera.unobserve_all()
    elif(x==1):
        GPIO.output(Start,1)
        GPIO.output(Recording,0)
        camera.unobserve_all()
i=i+1        

