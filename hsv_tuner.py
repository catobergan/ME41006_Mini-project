from flask import Flask, Response, render_template
import cv2
import numpy as np
from threading import Thread
import time

app = Flask(__name__)

# Global variables for HSV values
hsv_lower = np.array([90, 15, 250])
hsv_upper = np.array([102, 142, 255])
camera = None
output_frame = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(1)
        camera.set(3, 640)
        camera.set(4, 480)
    return camera

def process_frame():
    global output_frame, hsv_lower, hsv_upper
    
    while True:
        camera = get_camera()
        ret, frame = camera.read()
        if not ret:
            continue

        # Create copies for different displays
        original = frame.copy()
        
        # Process frame
        blurred = cv2.GaussianBlur(frame, (5, 5), 2)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # Clean up mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # Draw contours and centers on original frame
        contour_frame = frame.copy()
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
        
        # Find and draw center points for each contour
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Draw center point
                    cv2.circle(contour_frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot
                    cv2.circle(contour_frame, (cx, cy), 7, (255, 255, 255), 2)  # White outline
                    # Add coordinates text
                    cv2.putText(contour_frame, "({},{})".format(cx, cy), (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Combine frames for display
        top_row = np.hstack((original, contour_frame))
        bottom_row = np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 
                              cv2.bitwise_and(frame, frame, mask=mask)))
        output_frame = np.vstack((top_row, bottom_row))
        
        # Add labels
        labels = ['Original', 'Contours', 'Mask', 'Masked Result']
        h, w = output_frame.shape[:2]
        half_h, half_w = h//2, w//2
        
        for i, label in enumerate(labels):
            x = (i % 2) * half_w + 10
            y = (i // 2) * half_h + 30
            cv2.putText(output_frame, label, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def generate():
    global output_frame
    while True:
        if output_frame is None:
            continue
        
        # Encode the frame in JPEG format
        (flag, encoded_image) = cv2.imencode('.jpg', output_frame)
        if not flag:
            continue
        
        # Yield the output frame in byte format
        yield(b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/')
def index():
    return render_template('hsv_tuner.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_hsv/<values>')
def update_hsv(values):
    global hsv_lower, hsv_upper
    h_min, s_min, v_min, h_max, s_max, v_max = map(int, values.split(','))
    hsv_lower = np.array([h_min, s_min, v_min])
    hsv_upper = np.array([h_max, s_max, v_max])
    return 'OK'

if __name__ == '__main__':
    # Start the video processing thread
    Thread(target=process_frame, daemon=True).start()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)