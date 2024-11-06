import cv2
import numpy as np
from pathlib import Path

def extract_eye_region(image, eye_center, eye_width):
    """Extracts a square region around the eye from an image, resizes, normalizes, and reshapes it for model input.

    This function takes an image and the eye's center coordinates and width, extracts a square region centered
    on the eye, resizes it to 64x64 pixels, converts it to grayscale, normalizes it, and reshapes it for 
    input to a machine learning model.

    Args:
        image (numpy.ndarray): The input image in which the eye region is to be extracted.
        eye_center (tuple): A tuple (x, y) indicating the center coordinates of the eye in the image.
        eye_width (float): The width of the eye in pixels, used to calculate the extraction region size.

    Returns:
        numpy.ndarray or None: A 4D array with shape (1, 64, 64, 1) representing the processed eye region, ready for model input. 
        Returns None if the eye region is empty or out of bounds.
    """
    eye_size = int(eye_width * 1.5)  # Make the eye region slightly larger than the width
    x = int(eye_center[0] - eye_size // 2)
    y = int(eye_center[1] - eye_size // 2)
    eye_region = image[y:y+eye_size, x:x+eye_size]
    if eye_region.all == 0:
        return None
    if eye_region.shape[0] < 1 or eye_region.shape[1] < 1:
        return None
        
    resized_eye = cv2.resize(eye_region, (64, 64))
    gray_eye = cv2.cvtColor(resized_eye, cv2.COLOR_BGR2GRAY)
    normalized_eye = gray_eye / 255.0
    model_input = normalized_eye.reshape(1, 64, 64, 1)
    
    return model_input

def find_eyes(image, detector, verbose:bool=True):
    """Detects eyes in a face within an image, extracts eye regions, and optionally displays visual indicators.

    This function detects faces in the input image using a provided face detector, then identifies and extracts
    regions around the left and right eyes based on facial landmarks. Optionally, it draws bounding boxes around 
    detected faces and circles at eye locations for display.

    Args:
        image (numpy.ndarray): The input image in which to detect faces and eyes.
        detector (object): A face detector object with a `detect` method that returns faces with bounding boxes
            and landmarks.
        verbose (bool): If True, draws bounding boxes and eye landmarks on the image. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The image with optional bounding boxes and eye landmarks (if verbose is True).
            - numpy.ndarray or None: Extracted left eye region ready for further processing, or None if not found.
            - numpy.ndarray or None: Extracted right eye region ready for further processing, or None if not found.
    """
    # Detect faces
    _, faces = detector.detect(image)
    left_eye_img = None
    right_eye_img = None
    if faces is not None:
        for face in faces:
            # Extract bounding box
            box = face[0:4].astype(np.int32)
            
            # Extract eye landmarks
            landmarks = face[4:14].astype(np.int32).reshape((5,2))
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            eye_width = np.linalg.norm(right_eye-left_eye) / 2
            if left_eye is not None and right_eye is not None:
                left_eye_img = extract_eye_region(image=image, eye_center=left_eye, eye_width=eye_width)
                right_eye_img = extract_eye_region(image=image, eye_center=right_eye, eye_width=eye_width)
                     
                if verbose:
                    # Draw bounding box display
                    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                    # Draw eye positions for display
                    cv2.circle(img = image, center = left_eye, radius = 3, color = (255, 0, 0), thickness = -1)
                    cv2.circle(img = image, center = right_eye, radius = 3, color = (255, 0, 0), thickness = -1)
            
            
    return image, left_eye_img, right_eye_img
    