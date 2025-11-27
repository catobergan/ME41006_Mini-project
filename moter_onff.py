import RPi.GPIO as GPIO
import time

IN1 =24
IN2 =23
ENA =18

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

def motor_on():
	GPIO.output(IN1,GPIO.HIGH)
	GPIO.output(ENA,GPIO.HIGH)
	GPIO.output(IN2,GPIO.LOW)
	print("OPEN")
def motor_off():
	GPIO.output(IN1, GPIO.LOW)
	GPIO.output(IN2,GPIO.LOW)
	GPIO.output(ENA,GPIO.LOW)
	
	print("CLOSE")
try:
	while True:
		motor_on()
		time.sleep(0.5)
		motor_off()
		time.sleep(15)
		
except KeyboardInterrupt:
	print("over")
	GPIO.cleanup()
