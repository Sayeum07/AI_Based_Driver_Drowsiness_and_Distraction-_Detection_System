import os
import platform
import math
from datetime import datetime, timedelta
import logging
import random
import traceback
import atexit
import scipy.spatial.distance as scipy_dist  # For consistency with the drowsiness detection code
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, Response, jsonify, request, session
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables before other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import threading
import torch
# Removed: from ultralytics import YOLO
from collections import deque
from scipy.spatial import distance as dist
import dlib
import mediapipe as mp
import imutils
from imutils import face_utils
from collections import Counter
import json

# Force OpenCV to use a different backend
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = '0'

# Try different OpenCV backends in case default fails
OPENCV_BACKENDS = [
    cv2.CAP_MSMF,    # Microsoft Media Foundation (default on Windows)
    cv2.CAP_DSHOW,   # DirectShow (alternative on Windows)
    cv2.CAP_ANY      # Auto-detect
]

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Global variables - separate variables for drowsiness and distraction cameras
drowsiness_frame = None
distraction_frame = None
drowsiness_status = "Initializing..."
distraction_status = "Initializing..."
drowsiness_camera_available = False
distraction_camera_available = False
drowsiness_camera_lock = threading.Lock()
distraction_camera_lock = threading.Lock()
using_simulated_feed = False

# Initialize dlib's face detector and facial landmark predictor
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("✓ Facial landmark predictor loaded")

# Initialize MediaPipe solutions
print("Loading MediaPipe solutions...")
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe models with appropriate settings
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✓ MediaPipe solutions loaded")

# Define indices for facial landmarks around the eyes and mouth
# For dlib's 68-point facial landmark detector:
LEFT_EYE_START = 36
LEFT_EYE_END = 41
RIGHT_EYE_START = 42
RIGHT_EYE_END = 47
MOUTH_START = 48
MOUTH_END = 68

# Integration of functions from final-integration.py and main_dlib.py
def euclideanDist(a, b):
    """Calculate Euclidean distance between two points"""
    return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

def ear(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    return ((euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / 
            (2.0 * euclideanDist(eye[0], eye[3])))

def yawn(mouth):
    """Calculate mouth opening ratio for yawn detection"""
    return ((euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / 
            (2.0 * euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    """Determine face/head direction using 3D pose estimation"""
    try:
        image_points = np.array([
            (shape[33].x, shape[33].y),    # Nose tip
            (shape[8].x, shape[8].y),      # Chin
            (shape[45].x, shape[45].y),    # Left eye left corner
            (shape[36].x, shape[36].y),    # Right eye right corner
            (shape[54].x, shape[54].y),    # Left Mouth corner
            (shape[48].x, shape[48].y)     # Right mouth corner
        ], dtype="double")
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return translation_vector[1][0]
    except Exception as e:
        print(f"Error in getFaceDirection: {e}")
        return 0.0  # Default value if unable to determine face direction

# Constants for drowsiness detection
DROWSINESS_EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold
YAWN_THRESHOLD = 0.6  # Yawn detection threshold
FRAME_THRESHOLD_1 = 15  # Normal drowsiness frames
FRAME_THRESHOLD_2 = 10  # Body posture drowsiness frames
FRAME_THRESHOLD_3 = 5   # Drowsy after yawn frames

# Global variables for drowsiness detection state
drowsiness_flag = 0
yawn_countdown = 0

# Load distraction detection model
try:
    print("Loading distraction detection model...")
    # First check if the model directory exists
    if os.path.isdir('MobileNet_3Layers.h5'):
        # If it's a directory, load the SavedModel
        distraction_model = tf.saved_model.load('MobileNet_3Layers.h5')
    elif os.path.isfile('MobileNet_3Layers.h5'):
        # If it's a file, it might be a different format
        # Let's try an alternative approach
        try:
            # Try to load as a TensorFlow saved model
            distraction_model_dir = os.path.dirname('MobileNet_3Layers.h5')
            distraction_model = tf.keras.models.load_model(distraction_model_dir)
        except:
            # If that fails, try to load as a TFLite model
            interpreter = tf.lite.Interpreter(model_path='MobileNet_3Layers.h5')
            interpreter.allocate_tensors()
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            distraction_model = interpreter  # Store the interpreter as our model
    print("✓ Distraction detection model loaded")
except Exception as e:
    print(f"Failed to load distraction detection model: {e}")
    distraction_model = None

# Distraction detection classes
distraction_classes = {
    0: "Safe Driving",
    1: "Talking on Phone",
    2: "Texting Phone",
    3: "Turning",
    4: "Other Activities"  # Added for when hand is detected near face
}

# Load models
try:
    print("Loading drowsiness detection model...")
    custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}
    drowsiness_model = tf.keras.models.load_model('resnet50_fine_tune.h5', 
                                                  custom_objects=custom_objects,
                                                  compile=False)
    print("✓ Drowsiness model loaded")
    drowsiness_model.summary()
except Exception as e:
    print(f"Error loading drowsiness model: {e}")
    drowsiness_model = None

# This function will create simulated camera frames as a fallback
def create_simulated_frame(camera_type="drowsiness", frame_number=0):
    """Create a simulated camera frame for demo purposes"""
    # Create a black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add gradient background (blue for drowsiness, green for distraction)
    if camera_type == "drowsiness":
        for i in range(frame.shape[0]):
            blue_val = int(50 + (i / frame.shape[0]) * 100)
            frame[i, :] = (blue_val, 20, 0)  # BGR format
    else:
        for i in range(frame.shape[0]):
            green_val = int(50 + (i / frame.shape[0]) * 100)
            frame[i, :] = (20, green_val, 0)  # BGR format
    
    # Add camera title and simulated indicator
    cv2.putText(frame, f"{camera_type.capitalize()} Camera (SIMULATED)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add frame counter for debugging
    cv2.putText(frame, f"Frame: {frame_number}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Draw a simulated face (moving slightly)
    center_x = 320 + int(20 * np.sin(frame_number * 0.05))
    center_y = 240 + int(10 * np.cos(frame_number * 0.05))
    
    # Draw face
    cv2.circle(frame, (center_x, center_y), 70, (0, 200, 255), -1)
    
    # Determine eye status
    if camera_type == "drowsiness":
        eye_status = "open" if frame_number % 100 < 85 else "closed"
    else:
        eye_status = "open"  # Always open for distraction camera
    
    left_eye_y = center_y - 20
    right_eye_y = center_y - 20
    
    if eye_status == "open":
        # Open eyes
        cv2.circle(frame, (center_x - 25, left_eye_y), 10, (255, 255, 255), -1)
        cv2.circle(frame, (center_x + 25, right_eye_y), 10, (255, 255, 255), -1)
        cv2.circle(frame, (center_x - 25, left_eye_y), 4, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + 25, right_eye_y), 4, (0, 0, 0), -1)
    else:
        # Closed eyes
        cv2.line(frame, (center_x - 35, left_eye_y), (center_x - 15, left_eye_y), (0, 0, 0), 3)
        cv2.line(frame, (center_x + 15, right_eye_y), (center_x + 35, right_eye_y), (0, 0, 0), 3)
    
    # Draw mouth (smiling)
    smile_y = 15 if eye_status == "open" else 5
    cv2.ellipse(frame, (center_x, center_y + 20), (25, smile_y), 0, 0, 180, (0, 0, 0), 3)
    
    # For distraction camera, occasionally draw a phone
    is_distracted = camera_type == "distraction" and (frame_number % 150) > 100
    
    if is_distracted:
        # Draw phone
        phone_x = center_x + 70
        phone_y = center_y + 20
        cv2.rectangle(frame, (phone_x - 15, phone_y - 30), (phone_x + 15, phone_y + 30), (50, 50, 200), -1)
        cv2.rectangle(frame, (phone_x - 12, phone_y - 27), (phone_x + 12, phone_y - 5), (200, 200, 255), -1)
    
    # Determine status based on eye state and distraction
    if camera_type == "drowsiness":
        if eye_status == "closed":
            status = "Drowsy"
            status_color = (0, 0, 255)  # Red (BGR)
        else:
            status = "Alert"
            status_color = (0, 255, 0)  # Green (BGR)
    else:  # distraction camera
        if is_distracted:
            status = "Texting - Right"
            status_color = (0, 0, 255)  # Red (BGR)
        else:
            status = "Safe Driving"
            status_color = (0, 255, 0)  # Green (BGR)
    
    # Draw status box
    cv2.rectangle(frame, (10, 400), (300, 460), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 400), (300, 460), (255, 255, 255), 1)  # White border
    cv2.putText(frame, f"Status: {status}", (20, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Add LIVE indicator in corner
    cv2.circle(frame, (frame.shape[1] - 20, 20), 5, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (frame.shape[1] - 60, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame, status

# Function to check if cameras are available
def check_available_cameras():
    """Check for available cameras on the system"""
    global drowsiness_camera_available, distraction_camera_available, using_simulated_feed
    
    camera_list = []
    drowsiness_camera_index = None
    distraction_camera_index = None
    drowsiness_camera_backend = cv2.CAP_ANY
    distraction_camera_backend = cv2.CAP_ANY
    
    print("\n=== Camera Detection Started ===")
    
    # List of backends to try
    backends = [
        {"name": "Default", "id": cv2.CAP_ANY},
        {"name": "DirectShow", "id": cv2.CAP_DSHOW},
        {"name": "MSMF", "id": cv2.CAP_MSMF},
        {"name": "V4L2", "id": cv2.CAP_V4L2}
    ]
    
    # Try up to 4 camera indices with each backend
    for idx in range(4):
        for backend in backends:
            print(f"Trying camera index {idx} with backend {backend['name']}...")
            try:
                cap = cv2.VideoCapture(idx, backend["id"])
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    
                    # Check if we actually got a frame
                    if ret and frame is not None and frame.size > 0:
                        camera_info = {
                            "index": idx,
                            "backend": backend,
                            "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                        }
                        camera_list.append(camera_info)
                        
                        print(f"✅ Camera {idx} is available with backend {backend['name']}")
                        print(f"   Resolution: {camera_info['resolution']}")
                        
                        # Assign to drowsiness camera if not already assigned
                        if drowsiness_camera_index is None:
                            drowsiness_camera_index = idx
                            drowsiness_camera_backend = backend["id"]
                            drowsiness_camera_available = True
                        # Assign to distraction camera if it's a different index
                        elif distraction_camera_index is None and idx != drowsiness_camera_index:
                            distraction_camera_index = idx
                            distraction_camera_backend = backend["id"]
                            distraction_camera_available = True
                    else:
                        print(f"❌ Camera {idx} opened but couldn't capture frames with backend {backend['name']}")
                
                cap.release()
            except Exception as e:
                print(f"❌ Error testing camera {idx} with backend {backend['name']}: {str(e)}")
    
    # If no cameras were found, enable simulated mode
    if len(camera_list) == 0:
        print("\n❌ No working cameras detected. Using simulated mode.")
        using_simulated_feed = True
    else:
        print(f"\n✅ Found {len(camera_list)} working camera(s)")
        
        # If we found only one camera, use it for both
        if len(camera_list) == 1 and drowsiness_camera_index is not None and distraction_camera_index is None:
            print("Using the same camera for both drowsiness and distraction detection")
            distraction_camera_index = drowsiness_camera_index
            distraction_camera_backend = drowsiness_camera_backend
            distraction_camera_available = True
    
    print(f"Drowsiness camera: {'Available' if drowsiness_camera_available else 'Not available'}")
    print(f"Distraction camera: {'Available' if distraction_camera_available else 'Not available'}")
    print(f"Using simulated feed: {'Yes' if using_simulated_feed else 'No'}")
    
    # Return information about found cameras
    return {
        "camera_list": camera_list,
        "drowsiness_camera": drowsiness_camera_index if drowsiness_camera_available else 0,
        "distraction_camera": distraction_camera_index if distraction_camera_available else 0,
        "drowsiness_backend": drowsiness_camera_backend,
        "distraction_backend": distraction_camera_backend
    }

def process_drowsiness_camera(camera_index, backend, face_detector, q, stop_event):
    """Process frames from the drowsiness detection camera"""
    global drowsiness_frame, drowsiness_status, drowsiness_camera_available
    
    # Try to initialize camera
    cap = None
    frame_count = 0
    consecutive_failures = 0
    max_failures = 30  # Only try 30 times before switching to simulated mode
    
    try:
        if drowsiness_camera_available:
            print(f"\nInitializing drowsiness camera (index {camera_index})...")
            cap = cv2.VideoCapture(camera_index, backend)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                print(f"✗ Failed to open drowsiness camera at index {camera_index}")
                drowsiness_camera_available = False
            else:
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"✓ Successfully opened drowsiness camera at index {camera_index}")
        
        # Main camera processing loop
        while not stop_event.is_set():
            if drowsiness_camera_available and cap is not None and cap.isOpened():
                # Read a frame
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures < 5:  # Only show first few errors to reduce spam
                        print(f"Failed to grab frame from drowsiness camera, trying again... ({consecutive_failures}/{max_failures})")
                    
                    # If we've failed too many times, switch to simulated mode
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive failures on drowsiness camera, switching to simulated mode")
                        drowsiness_camera_available = False
                        if cap is not None:
                            cap.release()
                            cap = None
                        
                    # Sleep a bit before trying again
                    time.sleep(0.1)
                    continue
                
                # Success - reset failure counter
                consecutive_failures = 0
                
                # Increment frame counter
                frame_count += 1
                
                # Process frame for drowsiness detection
                fps = 30  # Just an estimate
                processed_frame, drowsiness_status, _ = process_frame(frame, face_detector, fps, mode="drowsiness")
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                drowsiness_frame = buffer.tobytes()
                
            else:
                # Use simulated feed if camera not available
                frame = create_simulated_frame("drowsiness", frame_count)
                frame_count += 1
                
                # Process simulated frame
                fps = 30
                processed_frame, drowsiness_status, _ = process_frame(frame, face_detector, fps, mode="drowsiness")
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                drowsiness_frame = buffer.tobytes()
                
                # Use simulated results
                if drowsiness_status == "Unknown":
                    drowsiness_status = random.choice(["Alert", "Normal", "Drowsy"] + ["Alert"] * 10)  # Bias toward Alert
                
                # Slower frame rate for simulated feed
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Error in drowsiness camera thread: {str(e)}")
        traceback.print_exc()
        drowsiness_camera_available = False
    
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        print("Drowsiness camera thread stopped")
        
        # Make sure we always have a valid frame
        if drowsiness_frame is None:
            frame = create_simulated_frame("drowsiness", 0)
            _, buffer = cv2.imencode('.jpg', frame)
            drowsiness_frame = buffer.tobytes()

def process_distraction_camera(camera_index, backend, face_detector, q, stop_event):
    """Process frames from the distraction detection camera"""
    global distraction_frame, distraction_status, distraction_camera_available
    
    # Try to initialize camera
    cap = None
    frame_count = 0
    consecutive_failures = 0
    max_failures = 30  # Only try 30 times before switching to simulated mode
    
    try:
        if distraction_camera_available:
            print(f"\nInitializing distraction camera (index {camera_index})...")
            cap = cv2.VideoCapture(camera_index, backend)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                print(f"✗ Failed to open distraction camera at index {camera_index}")
                distraction_camera_available = False
            else:
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"✓ Successfully opened distraction camera at index {camera_index}")
        
        # Main camera processing loop
        while not stop_event.is_set():
            if distraction_camera_available and cap is not None and cap.isOpened():
                # Read a frame
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures < 5:  # Only show first few errors to reduce spam
                        print(f"Failed to grab frame from distraction camera, trying again... ({consecutive_failures}/{max_failures})")
                    
                    # If we've failed too many times, switch to simulated mode
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive failures on distraction camera, switching to simulated mode")
                        distraction_camera_available = False
                        if cap is not None:
                            cap.release()
                            cap = None
                        
                    # Sleep a bit before trying again
                    time.sleep(0.1)
                    continue
                
                # Success - reset failure counter
                consecutive_failures = 0
                
                # Increment frame counter
                frame_count += 1
                
                # Process frame for distraction detection
                fps = 30  # Just an estimate
                processed_frame, _, distraction_status = process_frame(frame, face_detector, fps, mode="distraction")
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                distraction_frame = buffer.tobytes()
                
            else:
                # Use simulated feed if camera not available
                frame = create_simulated_frame("distraction", frame_count)
                frame_count += 1
                
                # Process simulated frame
                fps = 30
                processed_frame, _, distraction_status = process_frame(frame, face_detector, fps, mode="distraction")
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                distraction_frame = buffer.tobytes()
                
                # Use simulated results
                if distraction_status == "Unknown":
                    distraction_status = random.choice(["Attentive", "Normal", "Looking Away"] + ["Attentive"] * 10)  # Bias toward Attentive
                
                # Slower frame rate for simulated feed
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Error in distraction camera thread: {str(e)}")
        traceback.print_exc()
        distraction_camera_available = False
    
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        print("Distraction camera thread stopped")
        
        # Make sure we always have a valid frame
        if distraction_frame is None:
            frame = create_simulated_frame("distraction", 0)
            _, buffer = cv2.imencode('.jpg', frame)
            distraction_frame = buffer.tobytes()

# Update the process_frame function with enhanced drowsiness detection
def process_frame(frame, face_detector, fps, mode="both"):
    """Process a single frame for drowsiness/distraction detection"""
    global drowsiness_flag, yawn_countdown, is_alert_playing, last_alert_time, is_distraction_alert_playing, last_distraction_alert_time, pending_gps_email, last_email_time
    
    drowsiness_result = "Unknown"
    distraction_result = "Unknown"
    
    try:
        # Get frame dimensions
        frame_size = frame.shape
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using dlib
        faces = face_detector(gray, 0)
        
        # Add processing timestamp
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process face if detected
        if len(faces) > 0:
            face = faces[0]  # Use the largest face
            
            # Get face coordinates
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get facial landmarks for the face
            shape = predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)  # Convert to numpy array
            
            # Extract eye and mouth regions for drowsiness detection
            if mode == "drowsiness" or mode == "both":
                # Get eye landmarks
                leftEye = shape_np[LEFT_EYE_START:LEFT_EYE_END+1]
                rightEye = shape_np[RIGHT_EYE_START:RIGHT_EYE_END+1]
                mouth = shape_np[MOUTH_START:MOUTH_END+1]
                
                # Create hulls for visualization
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                
                # Calculate Eye Aspect Ratio (EAR)
                leftEAR = ear(leftEye)
                rightEAR = ear(rightEye)
                
                # Average the EAR of both eyes
                avgEAR = (leftEAR + rightEAR) / 2.0
                
                # Calculate yawn ratio
                yawnRatio = yawn(mouth)
                
                # Draw eye and mouth contours
                eyeContourColor = (255, 255, 255)  # Default white
                cv2.drawContours(frame, [leftEyeHull], -1, eyeContourColor, 1)
                cv2.drawContours(frame, [rightEyeHull], -1, eyeContourColor, 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)
                
                # Add EAR and yawn values to frame for debugging
                cv2.putText(frame, f"EAR: {avgEAR:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawn: {yawnRatio:.2f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detect yawn
                if yawnRatio > YAWN_THRESHOLD:
                    cv2.putText(frame, "Yawn Detected", (frame.shape[1] - 300, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    yawn_countdown = 1
                
                # Enhanced drowsiness detection logic based on EAR, yawn, and face direction
                if avgEAR < DROWSINESS_EAR_THRESHOLD:
                    drowsiness_flag += 1
                    eyeContourColor = (0, 255, 255)  # Yellow
                    
                    # Get face direction
                    face_direction = getFaceDirection(shape, frame_size)
                    
                    # Check if we should play an alert (with cooldown)
                    current_time = time.time()
                    should_play_alert = False
                    
                    # Check different drowsiness conditions
                    if yawn_countdown and drowsiness_flag >= FRAME_THRESHOLD_3:
                        eyeContourColor = (147, 20, 255)  # Purple
                        cv2.putText(frame, "Drowsy after yawn", (frame.shape[1] - 300, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        drowsiness_result = "Drowsy (After Yawn)"
                        should_play_alert = True
                        
                    elif drowsiness_flag >= FRAME_THRESHOLD_2 and face_direction < 0:
                        eyeContourColor = (255, 0, 0)  # Blue
                        cv2.putText(frame, "Drowsy (Body Posture)", (frame.shape[1] - 300, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        drowsiness_result = "Drowsy (Posture)"
                        should_play_alert = True
                        
                    elif drowsiness_flag >= FRAME_THRESHOLD_1:
                        eyeContourColor = (0, 0, 255)  # Red
                        cv2.putText(frame, "Drowsy (Normal)", (frame.shape[1] - 300, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        drowsiness_result = "Drowsy"
                        should_play_alert = True
                    else:
                        drowsiness_result = "Alert"
                    
                    # Play alert if needed and not on cooldown
                    if should_play_alert and not is_alert_playing and (current_time - last_alert_time) > alert_cooldown:
                        play_alert_sound()
                        is_alert_playing = True
                        last_alert_time = current_time
                        log_incident("Drowsiness", "Driver detected as drowsy")
                        
                    # Redraw eye contours with updated color
                    cv2.drawContours(frame, [leftEyeHull], -1, eyeContourColor, 2)
                    cv2.drawContours(frame, [rightEyeHull], -1, eyeContourColor, 2)
                
                elif avgEAR > DROWSINESS_EAR_THRESHOLD and drowsiness_flag:
                    # Reset flags when eyes open
                    yawn_countdown = 0
                    drowsiness_flag = 0
                    is_alert_playing = False
                    drowsiness_result = "Alert"
                else:
                    drowsiness_result = "Alert"
                    is_alert_playing = False
            
            # Process for distraction detection
            if mode == "distraction" or mode == "both":
                # Add hand detection using MediaPipe Hands
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb_frame)
                
                # Check if hands are detected
                hand_near_face = False
                hand_positions = []
                
                # Process hands if detected
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Get hand position (use wrist as reference point)
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                        hand_positions.append((wrist_x, wrist_y))
                        
                        # Check if hand is near face
                        if x1 - 50 <= wrist_x <= x2 + 50 and y1 - 50 <= wrist_y <= y2 + 50:
                            hand_near_face = True
                
                # Add warning message if hand is near face
                if hand_near_face:
                    distraction_result = "Talking on Phone."
                    cv2.putText(frame, "HOLD YOUR STEERING WHEEL!", (frame.shape[1] - 400, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # Add flashing warning
                    if int(time.time() * 2) % 2 == 0:  # Flash effect (2Hz)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
                    
                    # Play distraction alert sound with cooldown
                    current_time = time.time()
                    if not is_distraction_alert_playing and (current_time - last_distraction_alert_time) > alert_cooldown:
                        play_distraction_alert()
                        is_distraction_alert_playing = True
                        last_distraction_alert_time = current_time
                        log_incident("Distraction", "Driver Talking on Phone.")
                
                # Continue with existing distraction detection if no hand near face
                elif not hand_near_face:
                    # Reset distraction alert flag
                    is_distraction_alert_playing = False
                    
                    # Prepare image for distraction detection model
                    if distraction_model is not None:
                        try:
                            # Crop the face region with some margin
                            margin = 50
                            face_x1 = max(0, x1 - margin)
                            face_y1 = max(0, y1 - margin)
                            face_x2 = min(frame.shape[1], x2 + margin)
                            face_y2 = min(frame.shape[0], y2 + margin)
                            
                            # Make sure we have a valid crop region
                            if face_x1 < face_x2 and face_y1 < face_y2:
                                face_img = frame[face_y1:face_y2, face_x1:face_x2]
                                
                                # Resize to model input size
                                face_img = cv2.resize(face_img, (224, 224))
                                face_img = face_img / 255.0  # Normalize
                                
                                # Different processing based on model type
                                if isinstance(distraction_model, tf.lite.Interpreter):
                                    # TFLite model
                                    input_details = distraction_model.get_input_details()
                                    output_details = distraction_model.get_output_details()
                                    
                                    # Check input shape and adjust if needed
                                    input_shape = input_details[0]['shape']
                                    if len(input_shape) == 4:  # [batch, height, width, channels]
                                        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)
                                    
                                    # Set input tensor
                                    distraction_model.set_tensor(input_details[0]['index'], face_img)
                                    
                                    # Run inference
                                    distraction_model.invoke()
                                    
                                    # Get output
                                    predictions = distraction_model.get_tensor(output_details[0]['index'])
                                    pred_class = np.argmax(predictions[0])
                                    
                                else:
                                    # Regular TF/Keras model
                                    face_img = np.expand_dims(face_img, axis=0)
                                    
                                    try:
                                        # Try direct prediction
                                        predictions = distraction_model.predict(face_img)
                                        pred_class = np.argmax(predictions[0])
                                    except:
                                        try:
                                            # Try using signatures
                                            infer = distraction_model.signatures["serving_default"]
                                            input_tensor = tf.convert_to_tensor(face_img, dtype=tf.float32)
                                            predictions = infer(input_tensor)
                                            pred_class = np.argmax(list(predictions.values())[0])
                                        except:
                                            # Last resort: call directly if it's a callable model
                                            predictions = distraction_model(face_img)
                                            if isinstance(predictions, dict):
                                                pred_class = np.argmax(list(predictions.values())[0])
                                            else:
                                                pred_class = np.argmax(predictions[0])
                                
                                # Get prediction class
                                distraction_result = distraction_classes.get(pred_class, "Unknown")
                                
                                # Add distraction class to frame
                                cv2.putText(frame, f"Activity: {distraction_result}", 
                                            (frame.shape[1] - 300, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                            (0, 255, 0) if distraction_result == "Safe Driving" else (0, 0, 255), 2)
                                
                                # Play alert for distracted driving (anything except safe driving)
                                if distraction_result != "Safe Driving":
                                    current_time = time.time()
                                    if not is_distraction_alert_playing and (current_time - last_distraction_alert_time) > alert_cooldown:
                                        play_distraction_alert()
                                        is_distraction_alert_playing = True
                                        last_distraction_alert_time = current_time
                                        log_incident("Distraction", f"Driver detected {distraction_result}")
                                else:
                                    # Reset distraction alert when safe driving detected
                                    is_distraction_alert_playing = False
                        except Exception as e:
                            print(f"Error in distraction detection: {e}")
                            distraction_result = "Error"
                    else:
                        # Fallback to simple head pose estimation if model not available
                        # Get nose tip landmark (point 30 in dlib's 68-point model)
                        nose = shape.part(30)
                        
                        # Get facial width from left to right cheek
                        face_width = abs(shape.part(0).x - shape.part(16).x)
                        
                        # Calculate the position of the nose relative to the face center
                        face_center_x = (x1 + x2) // 2
                        relative_nose_pos = (nose.x - face_center_x) / (face_width / 2)
                        
                        # Draw nose position for visualization
                        cv2.circle(frame, (nose.x, nose.y), 5, (0, 255, 255), -1)
                        
                        # Add nose position value for debugging
                        cv2.putText(frame, f"Head Pos: {relative_nose_pos:.2f}", (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Determine distraction based on head pose
                        if relative_nose_pos < -0.2:
                            distraction_result = "Looking Left"
                            cv2.putText(frame, "LOOKING LEFT", (frame.shape[1] - 300, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        elif relative_nose_pos > 0.2:
                            distraction_result = "Looking Right" 
                            cv2.putText(frame, "LOOKING RIGHT", (frame.shape[1] - 300, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            distraction_result = "Safe Driving"
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mode == "drowsiness" or mode == "both":
                drowsiness_result = "No face detected"
                is_alert_playing = False
            if mode == "distraction" or mode == "both":
                # If no face is detected for distraction, it means driver is not looking straight
                distraction_result = "Turning."
                cv2.putText(frame, "Turning.", (frame.shape[1] - 400, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Add flashing warning
                if int(time.time() * 2) % 2 == 0:  # Flash effect (2Hz)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                
                # Play distraction alert sound with cooldown
                current_time = time.time()
                if not is_distraction_alert_playing and (current_time - last_distraction_alert_time) > alert_cooldown:
                    play_distraction_alert()
                    is_distraction_alert_playing = True
                    last_distraction_alert_time = current_time
                    log_incident("Distraction", "Turning.")
    
        # Add status overlays
        status_y_pos = frame.shape[0] - 40
        if mode == "drowsiness" or mode == "both":
            status_color = (0, 255, 0) if drowsiness_result == "Alert" else (0, 0, 255)
            cv2.putText(frame, f"Drowsiness: {drowsiness_result}", (10, status_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if mode == "distraction" or mode == "both":
            status_color = (0, 255, 0) if distraction_result == "Safe Driving" else (0, 0, 255)
            cv2.putText(frame, f"Distraction: {distraction_result}", (10, status_y_pos - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Set pending_gps_email if drowsiness > 4 seconds (fps*4)
        if drowsiness_flag >= int(fps*4) and not pending_gps_email and (time.time() - last_email_time > EMAIL_COOLDOWN):
            pending_gps_email = True
        
        return frame, drowsiness_result, distraction_result
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        cv2.putText(frame, f"Error: {str(e)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add status overlays even in case of error
        status_y_pos = frame.shape[0] - 40
        cv2.putText(frame, "Drowsiness: Error", (10, status_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Distraction: Error", (10, status_y_pos - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, "Error", "Error"

# Generate frames for the drowsiness detection feed
def generate_drowsiness_frames():
    """Generate frames from the drowsiness camera feed"""
    global drowsiness_frame
    
    # Create initial frame if none exists
    if drowsiness_frame is None:
        init_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(init_frame, "Initializing Drowsiness Camera...", (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', init_frame)
        drowsiness_frame = buffer.tobytes()
    
    while True:
        # Acquire lock before accessing the shared variable
        with drowsiness_camera_lock:
            if drowsiness_frame is None:
                # Create a default frame with text if no camera feed is available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Drowsiness Camera Unavailable", (120, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_data = jpeg.tobytes()
            else:
                frame_data = drowsiness_frame
        
        # Yield the frame in the format expected by Flask's Response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
        
        # Brief sleep to control frame rate
        time.sleep(0.04)  # ~25 fps

# Generate frames for the distraction detection feed
def generate_distraction_frames():
    """Generate frames from the distraction camera feed"""
    global distraction_frame
    
    # Create initial frame if none exists
    if distraction_frame is None:
        init_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(init_frame, "Initializing Distraction Camera...", (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', init_frame)
        distraction_frame = buffer.tobytes()
    
    while True:
        # Acquire lock before accessing the shared variable
        with distraction_camera_lock:
            if distraction_frame is None:
                # Create a default frame with text if no camera feed is available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Distraction Camera Unavailable", (120, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_data = jpeg.tobytes()
            else:
                frame_data = distraction_frame
        
        # Yield the frame in the format expected by Flask's Response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
        
        # Brief sleep to control frame rate
        time.sleep(0.04)  # ~25 fps

@app.route('/')
def home():
    profiles = load_profiles()
    return render_template('home.html', profiles=profiles)

@app.route('/drowsiness_feed')
def drowsiness_feed():
    """Video streaming route for drowsiness detection camera."""
    return Response(generate_drowsiness_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distraction_feed')
def distraction_feed():
    """Video streaming route for distraction detection camera."""
    return Response(generate_distraction_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    """Return the current status of both detection systems"""
    global drowsiness_status, distraction_status, drowsiness_camera_available, distraction_camera_available, using_simulated_feed, pending_gps_email
    
    return jsonify({
        'drowsiness': drowsiness_status if drowsiness_camera_available or using_simulated_feed else "Camera Unavailable",
        'distraction': distraction_status if distraction_camera_available or using_simulated_feed else "Camera Unavailable",
        'drowsiness_camera_available': drowsiness_camera_available,
        'distraction_camera_available': distraction_camera_available,
        'using_simulated_feed': using_simulated_feed,
        'pending_gps_email': pending_gps_email
    })

@app.route('/camera_test')
def camera_test():
    """Test camera connections and return status"""
    global drowsiness_camera_available, distraction_camera_available, using_simulated_feed
    
    # Check camera state
    camera_info = check_available_cameras()
    
    return jsonify({
        'available_cameras': len(camera_info['camera_list']),
        'camera_list': camera_info['camera_list'],
        'drowsiness_camera_available': drowsiness_camera_available,
        'distraction_camera_available': distraction_camera_available,
        'using_simulated_feed': using_simulated_feed
    })

@app.route('/toggle_monitoring', methods=['POST'])
def toggle_monitoring():
    """Toggle the monitoring state"""
    global drowsiness_camera_available, distraction_camera_available, using_simulated_feed
    
    # If no real cameras are available, enable simulated feeds
    if not drowsiness_camera_available and not distraction_camera_available:
        using_simulated_feed = True
    
    return jsonify({
        'status': 'active',
        'drowsiness_camera_available': drowsiness_camera_available,
        'distraction_camera_available': distraction_camera_available,
        'using_simulated_feed': using_simulated_feed
    })

@app.route('/dashboard')
def dashboard():
    if 'active_profile' not in session:
        return redirect('/')
    return render_template('index.html')

# Setup for audio alerts
def play_alert_sound():
    """Play the drowsiness alert sound"""
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.PlaySound('alert-sound.mp3', winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            # For Linux or Mac, use pygame if available
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load('alert-sound.mp3')
                pygame.mixer.music.play()
            except ImportError:
                # Fallback to os system commands
                if platform.system() == 'Darwin':  # macOS
                    os.system("afplay alert-sound.mp3 &")
                else:  # Linux
                    os.system("aplay alert-sound.mp3 &")
    except Exception as e:
        print(f"Failed to play alert sound: {e}")

def play_distraction_alert():
    """Play a different alert sound for distraction warnings"""
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.PlaySound('focus.mp3', winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            # For Linux or Mac, use pygame if available
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load('focus.mp3')
                pygame.mixer.music.play()
            except ImportError:
                # Fallback to os system commands
                if platform.system() == 'Darwin':  # macOS
                    os.system("afplay focus.mp3 &")
                else:  # Linux
                    os.system("aplay focus.mp3 &")
    except Exception as e:
        print(f"Failed to play distraction alert sound: {e}")

def play_break_sound():
    """Play the take a break sound"""
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.PlaySound('take_a_break.mp3', winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            # For Linux or Mac, use pygame if available
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load('take_a_break.mp3')
                pygame.mixer.music.play()
            except ImportError:
                # Fallback to os system commands
                if platform.system() == 'Darwin':  # macOS
                    os.system("afplay take_a_break.mp3 &")
                else:  # Linux
                    os.system("aplay take_a_break.mp3 &")
    except Exception as e:
        print(f"Failed to play break sound: {e}")

# Global variables for audio state
is_alert_playing = False
is_distraction_alert_playing = False
last_alert_time = 0
last_distraction_alert_time = 0
alert_cooldown = 3  # seconds between alerts

# Add new function for logging incidents
def log_incident(incident_type, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {incident_type}: {details}\n"
    with open("incidents.log", "a") as log_file:
        log_file.write(log_entry)

def get_analytics_data():
    """Process incident logs and return analytics data"""
    try:
        # Read the incidents log file
        with open("incidents.log", "r") as log_file:
            incidents = log_file.readlines()
        
        # Initialize counters
        drowsiness_count = 0
        distraction_count = 0
        distraction_types = Counter()
        hourly_distribution = Counter()
        
        # Get current time and calculate time periods
        current_time = datetime.now()
        current_period_start = current_time - timedelta(hours=24)  # Last 24 hours
        previous_period_start = current_period_start - timedelta(hours=24)  # Previous 24 hours
        
        # Initialize period counters
        current_period = {
            'drowsiness': 0,
            'distraction': 0,
            'total': 0
        }
        previous_period = {
            'drowsiness': 0,
            'distraction': 0,
            'total': 0
        }
        
        # Process each incident
        for incident in incidents:
            try:
                # Parse timestamp and details
                timestamp_str = incident.split(" - ")[0]
                incident_type = incident.split(" - ")[1].split(":")[0]
                details = incident.split(": ")[1].strip()
                
                # Convert timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                hour = timestamp.strftime("%H:00")
                
                # Update counters
                if incident_type == "Drowsiness":
                    drowsiness_count += 1
                    if timestamp >= current_period_start:
                        current_period['drowsiness'] += 1
                    elif timestamp >= previous_period_start:
                        previous_period['drowsiness'] += 1
                elif incident_type == "Distraction":
                    distraction_count += 1
                    distraction_types[details] += 1
                    if timestamp >= current_period_start:
                        current_period['distraction'] += 1
                    elif timestamp >= previous_period_start:
                        previous_period['distraction'] += 1
                
                if timestamp >= current_period_start:
                    current_period['total'] += 1
                elif timestamp >= previous_period_start:
                    previous_period['total'] += 1
                
                hourly_distribution[hour] += 1
                
            except Exception as e:
                print(f"Error processing incident: {e}")
                continue
        
        # Calculate statistics
        total_incidents = drowsiness_count + distraction_count
        drowsiness_percentage = (drowsiness_count / total_incidents * 100) if total_incidents > 0 else 0
        
        # Calculate trends
        def calculate_trend(current, previous):
            if previous == 0:
                return 0 if current == 0 else 100
            return round(((current - previous) / previous) * 100, 1)
        
        trends = {
            'total_trend': calculate_trend(current_period['total'], previous_period['total']),
            'drowsiness_trend': calculate_trend(current_period['drowsiness'], previous_period['drowsiness']),
            'distraction_trend': calculate_trend(current_period['distraction'], previous_period['distraction']),
            'percentage_trend': calculate_trend(
                (current_period['drowsiness'] / current_period['total'] * 100) if current_period['total'] > 0 else 0,
                (previous_period['drowsiness'] / previous_period['total'] * 100) if previous_period['total'] > 0 else 0
            )
        }
        
        # Prepare data for charts
        hourly_data = [{"hour": hour, "count": count} 
                      for hour, count in sorted(hourly_distribution.items())]
        
        distraction_data = [{"type": dist_type, "count": count} 
                          for dist_type, count in distraction_types.most_common()]
        
        return {
            "total_incidents": total_incidents,
            "drowsiness_count": drowsiness_count,
            "distraction_count": distraction_count,
            "drowsiness_percentage": round(drowsiness_percentage, 2),
            "hourly_distribution": hourly_data,
            "distraction_types": distraction_data,
            **trends  # Include all trend calculations
        }
    except Exception as e:
        print(f"Error generating analytics: {e}")
        return {
            "error": str(e),
            "total_incidents": 0,
            "drowsiness_count": 0,
            "distraction_count": 0,
            "drowsiness_percentage": 0,
            "hourly_distribution": [],
            "distraction_types": [],
            "total_trend": 0,
            "drowsiness_trend": 0,
            "distraction_trend": 0,
            "percentage_trend": 0
        }

@app.route('/analytics')
def analytics():
    """Render the analytics dashboard"""
    return render_template('analytics.html')

@app.route('/api/analytics')
def get_analytics():
    """API endpoint to get analytics data"""
    return jsonify(get_analytics_data())

@app.route('/live-data')
def get_live_data():
    """API endpoint to get real-time monitoring data"""
    try:
        # Read recent incidents from the log file
        with open("incidents.log", "r") as log_file:
            incidents = log_file.readlines()
        
        # Get the last 50 incidents
        recent_incidents = incidents[-50:] if len(incidents) > 50 else incidents
        
        # Process incidents into alerts
        alerts = []
        for incident in recent_incidents:
            try:
                # Parse timestamp and details
                timestamp_str = incident.split(" - ")[0]
                incident_type = incident.split(" - ")[1].split(":")[0]
                details = incident.split(": ")[1].strip()
                
                # Convert timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                # Create alert object
                alert = {
                    "id": f"{timestamp_str}_{incident_type}_{hash(details)}",
                    "timestamp": timestamp.isoformat(),
                    "type": incident_type,
                    "details": details,
                    "resolved_at": None  # Will be updated when the alert is resolved
                }
                
                # Check if this alert is resolved (if there's a subsequent "Safe Driving" or "Normal" status)
                if incident_type == "Distraction" and "Safe Driving" in details:
                    alert["resolved_at"] = timestamp.isoformat()
                elif incident_type == "Drowsiness" and "Normal" in details:
                    alert["resolved_at"] = timestamp.isoformat()
                
                alerts.append(alert)
            except Exception as e:
                print(f"Error processing incident for live data: {e}")
                continue
        
        return jsonify({
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error generating live data: {e}")
        return jsonify({
            "error": str(e),
            "alerts": [],
            "timestamp": datetime.now().isoformat()
        })

# Profile storage (in a real app, this would be a database)
PROFILES_FILE = 'profiles.json'

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=4)

@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    profiles = load_profiles()
    return jsonify(profiles)

@app.route('/api/profiles', methods=['POST'])
def create_profile():
    data = request.get_json()
    profiles = load_profiles()
    # Generate a unique key (e.g., lowercase name + number if needed)
    base_key = data['name'].strip().lower().replace(' ', '')
    key = base_key
    i = 1
    while key in profiles:
        key = f"{base_key}{i}"
        i += 1
    new_profile = {
        'name': data['name'],
        'email': data['email']
    }
    profiles[key] = new_profile
    save_profiles(profiles)
    return jsonify(new_profile), 201

@app.route('/api/select-profile', methods=['POST'])
def select_profile():
    data = request.get_json()
    profile_id = data.get('profile_id')
    
    profiles = load_profiles()
    selected_profile = next((p for p in profiles if p['id'] == profile_id), None)
    
    if selected_profile:
        session['active_profile'] = selected_profile
        return jsonify({'message': 'Profile selected successfully'})
    
    return jsonify({'error': 'Profile not found'}), 404

@app.route('/select_profile', methods=['POST'])
def select_profile_shortcut():
    data = request.get_json()
    profile_id = data.get('profile_id')
    profiles = load_profiles()
    if profile_id in profiles:
        session['active_profile'] = profiles[profile_id]
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid profile'})

# Update SMTP credentials
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "usernotfound.ai@gmail.com"
SMTP_PASSWORD = "woxplkdpgofiuqgi"

def get_address_from_latlon(lat, lon):
    try:
        url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1'
        headers = {'User-Agent': 'SafeOnix-Drowsiness-Alert/1.0'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            address = data.get('address', {})
            return {
                'full_address': data.get('display_name', ''),
                'city': address.get('city', address.get('town', address.get('village', ''))),
                'region': address.get('state', ''),
                'country': address.get('country', ''),
                'postcode': address.get('postcode', '')
            }
        else:
            return {}
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
        return {}

# Update send_emergency_email to send a styled HTML email matching the screenshot, with all fields and sections (alert, location, actions, Google Maps button).
def send_emergency_email(profile_email, lat, lon, address_info, accuracy):
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
        html_body = f"""
        <div style='font-family: Arial, sans-serif; background: #fff; padding: 0; margin: 0;'>
            <div style='background: #fff; padding: 0;'>
                <div style='background: #fff; padding: 0;'>
                    <div style='background: #fff; padding: 0;'>
                        <div style='border-left: 6px solid #e53935; background: #fff5f5; padding: 1.5rem 2rem 1.5rem 2rem; margin-bottom: 2rem;'>
                            <h2 style='color: #e53935; margin: 0 0 1rem 0;'><i style="margin-right:8px;" class="fas fa-exclamation-triangle"></i>EMERGENCY ALERT</h2>
                            <div style='font-size: 1.1rem; margin-bottom: 0.5rem;'><b>Type:</b> <span style='color: #5e35b1;'>Severe Drowsiness</span></div>
                            <div style='margin-bottom: 0.5rem;'><b>Details:</b> Driver showing severe signs of drowsiness.</div>
                            <div style='margin-bottom: 0.5rem;'><b>Time:</b> {now}</div>
                        </div>
                        <div style='background: #f7f7f7; padding: 1.5rem 2rem; border-radius: 8px; margin-bottom: 2rem;'>
                            <div style='font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;'>Location Details:</div>
                            <div><b>Latitude:</b> {lat}</div>
                            <div><b>Longitude:</b> {lon}</div>
                            <div><b>Accuracy:</b> {accuracy} meters</div>
                            <div style='margin-top: 0.5rem;'><b>Full Address:</b> {address_info.get('full_address', 'N/A')}</div>
                            <a href='{maps_link}' style='display:inline-block; margin-top:1.5rem; background:#43a047; color:#fff; padding:0.75rem 1.5rem; border-radius:5px; text-decoration:none; font-weight:bold; font-size:1rem;'>
                                View Location on Google Maps
                            </a>
                        </div>
                        <div style='background: #e3f2fd; padding: 1.5rem 2rem; border-radius: 8px; margin-bottom: 2rem;'>
                            <div style='font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;'>Recommended Actions:</div>
                            <ul style='margin:0; padding-left:1.2rem;'>
                                <li>Check on the driver immediately</li>
                                <li>Call emergency services if necessary</li>
                                <li>Monitor the situation closely</li>
                            </ul>
                        </div>
                        <div style='color: #888; font-size: 0.95rem; margin-top: 2rem;'>
                            This is an automated alert from the Driver Monitoring System.<br>
                            Please do not reply to this email.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_USERNAME
        msg['To'] = profile_email
        msg['Subject'] = "EMERGENCY ALERT: Driver Drowsiness Detected"
        msg.attach(MIMEText(html_body, 'html'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Store a flag and last sent time for drowsiness email
last_email_time = 0
EMAIL_COOLDOWN = 300  # 5 minutes
pending_gps_email = False

@app.route('/api/send-location', methods=['POST'])
def api_send_location():
    global last_email_time, pending_gps_email
    if 'active_profile' not in session:
        return jsonify({'success': False, 'error': 'No active profile'}), 400
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    accuracy = data.get('accuracy', 'N/A')
    if not lat or not lon:
        return jsonify({'success': False, 'error': 'Missing location'}), 400
    profile = session['active_profile']
    now = time.time()
    # Robust 5-minute cooldown enforcement
    if pending_gps_email and (now - last_email_time > EMAIL_COOLDOWN):
        pending_gps_email = False  # Set flag to False immediately to prevent multiple emails
        address_info = get_address_from_latlon(lat, lon)
        if send_emergency_email(profile['email'], lat, lon, address_info, accuracy):
            last_email_time = now
            return jsonify({'success': True, 'email': profile['email']})
        else:
            return jsonify({'success': False, 'error': 'Failed to send email'}), 500
    return jsonify({'success': False, 'error': 'Not ready or cooldown active'}), 429

def main():
    """Main function to start the application"""
    global drowsiness_camera_available, distraction_camera_available, using_simulated_feed, detector
    
    print("\n=== Driver Monitoring System ===")
    print("Starting application...")
    
    # Check for available cameras
    camera_info = check_available_cameras()
    
    # Create empty queue and stop event
    empty_queue = deque(maxlen=100)
    stop_event = threading.Event()
    
    # Start drowsiness detection with primary camera
    if drowsiness_camera_available or using_simulated_feed:
        drowsiness_thread = threading.Thread(
            target=process_drowsiness_camera,
            args=(
                camera_info.get("drowsiness_camera", 0), 
                camera_info.get("drowsiness_backend", cv2.CAP_ANY), 
                detector, 
                empty_queue,  # Empty queue, not used
                stop_event
            ),
            daemon=True
        )
        drowsiness_thread.start()
        print("Drowsiness detection thread started")
    
    # Start distraction detection with secondary camera
    if distraction_camera_available or using_simulated_feed:
        distraction_thread = threading.Thread(
            target=process_distraction_camera,
            args=(
                camera_info.get("distraction_camera", 0), 
                camera_info.get("distraction_backend", cv2.CAP_ANY), 
                detector, 
                empty_queue,  # Empty queue, not used
                stop_event
            ),
            daemon=True
        )
        distraction_thread.start()
        print("Distraction detection thread started")
    
    # Register cleanup function
    atexit.register(lambda: stop_event.set())
    
    # Run the Flask app
    print("\n=== Web Interface Ready ===")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', debug=False, threaded=True)
    
    # This will run when the app exits
    print("Shutting down camera threads...")
    stop_event.set()
    time.sleep(1)  # Give threads time to clean up

if __name__ == '__main__':
    main()