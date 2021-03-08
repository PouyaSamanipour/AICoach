import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
outputPin=12
Start=22
i=0
pushbutton=40
GPIO.setup(outputPin,GPIO.OUT)
GPIO.setup(Start,GPIO.OUT)
GPIO.setup(pushbutton,GPIO.IN)
GPIO.output(outputPin,1)
time.sleep(5)
GPIO.output(outputPin,0)
GPIO.output(Start,1)
time.sleep(5)
GPIO.output(Start,0)
while (i<=10):
    time.sleep(3)
    x=GPIO.input(pushbutton)
    print(x)
    i=i+1
    if (x==0):
        GPIO.output(Start,1)
    if (x==1):
         GPIO.output(Start,0)
GPIO.output(Start,0)
GPIO.cleanup()
