import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
outputPin=12
Start=22
Recording=36
pushbutton=15
i=0
GPIO.setup(outputPin,GPIO.OUT)
GPIO.setup(Start,GPIO.OUT)
GPIO.setup(pushbutton,GPIO.IN)
GPIO.setup(Recording,GPIO.OUT)
GPIO.output(outputPin,1)
time.sleep(5)
#GPIO.output(outputPin,0)
GPIO.output(Start,1)
time.sleep(5)
#GPIO.output(Start,0)
oldstate=True
while (i<=200):
    time.sleep(0.2)
    x=GPIO.input(pushbutton)
    print(x)
    i=i+1
    if (x==0):
        GPIO.output(Recording,1)
        time.sleep(0.5)
        #GPIO.output(Recording,0)
    elif (x==1):
         GPIO.output(Recording,0)
GPIO.output(Start,0)
GPIO.output(Recording,0)
GPIO.output(outputPin,0)
GPIO.cleanup()
