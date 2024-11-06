import cv2
from pathlib import Path
import utils as utils # Helper functions to project
from tensorflow.keras.models import load_model
import DDD_model
import time

# Initialize the webcam.
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# Load custom drowsiness detection and YuNet models.
DDD_model = DDD_model.load_model(load_weights=True)
base_path = Path(__file__).resolve().parent
model_path = base_path / "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(model_path, "", (int(width), int(height)))

eyes_closed_start = None
eyes_closed_duration = 0
# Adjust this to change the time it takes before system labels as drowsy.
ALERT_THRESHOLD = 1.0

fps = 0
frame_count = 0
start_time = time.time()

# This coin variable is used to jump between the different eyes to increase efficiency.

# coin = 2 results in both eyes being analysed each frame.
# coin = 1 results in one eye being alternatively analysed each frame.
coin = 2

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_with_boxes, left_eye, right_eye = utils.find_eyes(frame, detector, verbose=1)

    if left_eye is not None and right_eye is not None:
        if coin == 2:
            prediction_L = DDD_model.predict(left_eye, verbose=0)[0][0]
            prediction_R = DDD_model.predict(right_eye, verbose=0)[0][0]
        elif coin == 1:
            prediction_L = DDD_model.predict(left_eye, verbose=0)[0][0]
            prediction_R = prediction_L
            coin = 0
        else:
            prediction_R = DDD_model.predict(right_eye, verbose=0)[0][0]
            prediction_L = prediction_R
            coin = 1
                    
        if prediction_L > 0.5 and prediction_R > 0.5:
            eye_status = "Open"
            eyes_closed_start = None
            eyes_closed_duration = 0
        else:
            eye_status = "Closed"
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            else:
                eyes_closed_duration = time.time() - eyes_closed_start

        
    # Determine color of text based on the duration of eyes closed.
    else:
        eye_status = "Not Detected"
        if eyes_closed_start is None:
            eyes_closed_start = time.time()
        else:
            eyes_closed_duration = time.time() - eyes_closed_start
            
    if frame_with_boxes is not None:     
        cv2.putText(frame_with_boxes, f"Eyes: {eye_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        timer_color = (0, 0, 255) if eyes_closed_duration > ALERT_THRESHOLD else (0, 255, 0)
        cv2.putText(frame_with_boxes, f"Drowsy for: {eyes_closed_duration:.2f}s", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
      
        # Calculate and display FPS
        frame_count += 1
        if time.time() - start_time > 1:  # Update every second
            fps = frame_count / (time.time() - start_time)
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(frame_with_boxes, "Press 'q' to quit", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Webcam', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    elif frame_with_boxes is not None:
        cv2.imshow('Webcam', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
