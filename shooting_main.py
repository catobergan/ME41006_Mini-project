import cv2
import numpy as np
import Adafruit_PCA9685
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

# initialize the working encironment of the servo motor
pwm = Adafruit_PCA9685.PCA9685(0x41)
pwm.set_pwm_freq(50)

def set_servo_angle(channel, angle):
    angle = 4096 * ((angle * 11) + 500) / 20000
    pwm.set_pwm(channel, 0, int(angle))

# initialize the position of the motor
currentAngle_x = 90
currentAngle_y = 91 # Static y value is used, and thus needs to be tuned
set_servo_angle(1, currentAngle_x)
set_servo_angle(2, currentAngle_y)
set_servo_angle(3, 90)

cap = cv2.VideoCapture(0)

# HSV-values used for camera target detection
hsv_lower = np.array([40, 60, 180])
hsv_upper = np.array([100, 140, 255])

cap.set(3, 640)
cap.set(4, 480)

print("Press 's' to save image")
print("Press 'q' to exit program")
# Initialize the target center of the image
targetCenter = [320, 240]

# Tunable target offset from center
target_offset_x = 50  # Offset in x direction (pixels)
target_offset_y = 0  # Offset in y direction (pixels)

# P.controller gain
p_kx = 0.028 # critical value (X)
p_x = 0.5*p_kx # Proportional gain (X) from Ziegler-Nichols method
p_y = 0.00 # Proportional gain (Y) zero due to constant y value

# Shooting control
last_shot_time = 0.0
shoot_interval = 0.3 # seconds between shots while target is detected
motor_pulse = 0.1 # seconds to keep motor ON for each shot
error_threshold = 10 # Threshold for shooting (pixels)

while True:
    ret, frame = cap.read()
    original = frame.copy()
    frame = cv2.GaussianBlur(frame, (5, 5), 2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 2)
    res = cv2.bitwise_and(frame, frame, mask=mask)

	# center of image frame
    h, w = frame.shape[:2]
    cv2.line(res, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.line(res, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 1
    area_max = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1700:
            print("area:", area)
            cv2.drawContours(res, [c], contourIdx=-1, color=(255, 255, 255), thickness=5, lineType=cv2.LINE_AA)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(res, (cx, cy), 5, (255, 0, 0), -1)
            print("center:", cx, cy)
            
            if area > area_max:
                area_max = area
                targetCenter = [cx, cy]
            break

    # Compute new angles using PI control: u[k] = u[k-1] + g0*e[k] + g1*e[k-1]
    if area > 1700:
        # Calculate error using tunable offset
        error_x = targetCenter[0] - (w // 2 - target_offset_x) # invert if servo direction differs
        error_y = (h // 2 - target_offset_y) - targetCenter[1] # invert if servo direction differs
        
        # Calculate combined error magnitude
        error_magnitude = np.sqrt(error_x**2 + error_y**2)

		# P-controller
        delta_angle_x = p_x * error_x
        new_angle_x = currentAngle_x + delta_angle_x
        delta_angle_y = -(p_y * error_y)
        new_angle_y = currentAngle_y + delta_angle_y
        
        # Clip angles to valid servo range
        new_angle_x = np.clip(new_angle_x, 0, 180)
        new_angle_y = np.clip(new_angle_y, 0, 180)
        
        # Update servo positions
        set_servo_angle(1, new_angle_x)
        set_servo_angle(2, new_angle_y)
        print("Error: ", error_x)
        
        # Update state for next iteration
        currentAngle_x = new_angle_x
        currentAngle_y = new_angle_y
        
        # Shooting
        current_time = time.time()
        if error_magnitude < error_threshold:
            # shoot periodically while error remains below threshold
            if current_time - last_shot_time >= shoot_interval:
                print("SHOOTING! Target locked!")
                motor_on()
                time.sleep(motor_pulse)  # Motor on for configured pulse
                motor_off()
                last_shot_time = current_time
        else:
            last_shot_time = 0.0

    cv2.imshow("original", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite('original.jpg', original)
        cv2.imwrite('mask.jpg', mask)
        cv2.imwrite('result.jpg', res)
        print("Save!")
        break
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


