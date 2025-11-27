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
set_servo_angle(1, 90)
set_servo_angle(2, 90)
set_servo_angle(3, 90)

cap = cv2.VideoCapture(0)

hsv_lower = np.array([0, 0, 245])
hsv_upper = np.array([10, 10, 255])

cap.set(3, 640)
cap.set(4, 480)

print("Press 's' to save image")
print("Press 'q' to exit program")
# Initialize the target center of the object (e.g., [320, 240])
targetCenter = [320, 240]
currentAngle_x = 90
currentAngle_y = 91

# Tunable target offset from center
target_offset_x = 50  # Offset in x direction (pixels)
target_offset_y = 0  # Offset in y direction (pixels)

# PI Controller parameters: u[k] = u[k-1] + g0*e[k] + g1*e[k-1]
pi_g0_x = 0.009   # Proportional gain (X)
pi_g1_x = 0.003   # Integral gain (X) — tune to reduce steady-state error
pi_g0_y = 0.005   # Proportional gain (Y)
pi_g1_y = 0.002   # Integral gain (Y)

# PI state: previous errors
pi_prev_error_x = 0.0
pi_prev_error_y = 0.0

# Shooting control
# Shoot repeatedly while target is detected with a minimum interval between shots
last_shot_time = 0.0
shoot_interval = 0.3   # seconds between shots while target is detected
motor_pulse = 0.1      # seconds to keep motor ON for each shot
error_threshold = 20   # Threshold for shooting (pixels)

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

    h, w = frame.shape[:2]
    cv2.line(res, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.line(res, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 1
    area_max = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
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
    if area > 1000:
        # Calculate error using tunable offset
        error_x = targetCenter[0] - (w // 2 - target_offset_x)
        error_y = (h // 2 - target_offset_y) - targetCenter[1]  # invert if servo direction differs
        
        # Calculate combined error magnitude
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        # Apply PI control formula for X axis: u[k] = u[k-1] + g0*e[k] + g1*e[k-1]
        delta_angle_x = pi_g0_x * error_x + pi_g1_x * pi_prev_error_x
        new_angle_x = currentAngle_x + delta_angle_x
        
        # Apply PI control formula for Y axis (negate for correct direction)
        delta_angle_y = -(pi_g0_y * error_y + pi_g1_y * pi_prev_error_y)
        new_angle_y = currentAngle_y + delta_angle_y
        
        # Clip angles to valid servo range
        new_angle_x = np.clip(new_angle_x, 0, 180)
        new_angle_y = np.clip(new_angle_y, 0, 180)
        
        # Update servo positions
        set_servo_angle(1, new_angle_x)
        set_servo_angle(2, new_angle_y)
        
        print(f"Error magnitude: {error_magnitude:.2f} px, X error: {error_x:.2f}, Y error: {error_y:.2f}")
        print(f"Angles -> X: {new_angle_x:.2f}, Y: {new_angle_y:.2f}")
        
        # Update state for next iteration
        currentAngle_x = new_angle_x
        currentAngle_y = new_angle_y
        pi_prev_error_x = error_x
        pi_prev_error_y = error_y
        
        # Shoot when error is less than threshold (repeat every `shoot_interval` seconds)
        current_time = time.time()
        if error_magnitude < error_threshold:
            # fire periodically while the error remains below threshold
            if current_time - last_shot_time >= shoot_interval:
                print("SHOOTING! Target locked!")
                motor_on()
                time.sleep(motor_pulse)  # Motor on for configured pulse
                motor_off()
                last_shot_time = current_time
        else:
            # reset timer when target lost so we can shoot immediately when reacquired
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
