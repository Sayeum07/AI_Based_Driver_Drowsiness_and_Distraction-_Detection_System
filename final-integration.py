import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import winsound
import train as train
import sys, webbrowser, datetime
import time  # Add time module for delays

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

#EAR -> Eye Aspect ratio
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    cv2.imwrite('left-eye.jpg', img[y1:y2, x1:x2])
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    cv2.imwrite('right-eye.jpg', img[y1:y2, x1:x2])
# open_avg = train.getAvg()
# close_avg = train.getAvg()

# Replace VLC player with winsound functions
def play_alert():
    winsound.Beep(1000, 500)  # 1000 Hz for 500 milliseconds

def play_break_alert():
    winsound.Beep(2000, 1000)  # 2000 Hz for 1 second

frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5

close_thresh = 0.3#(close_avg+open_avg)/2.0
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1
redirect_threshold = 5  # Increased from 3 to 5
last_redirect_time = time.time()  # Track when the last redirect happened

# print(close_thresh)

capture = cv2.VideoCapture(0)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

is_alert_playing = False  # Track if alert is currently playing

while(True):
    ret, frame = capture.read()
    size = frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    rects = detector(gray, 0)
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # print("Mouth Open Ratio", yawn(shape[mStart:mEnd]))
        leftEAR = ear(leftEye) #Get the left eye aspect ratio
        rightEAR = ear(rightEye) #Get the right eye aspect ratio
        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)

        if(yawn(shape[mStart:mEnd])>0.6):
            cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            yawn_countdown=1

        if(avgEAR<close_thresh):
            flag+=1
            eyeContourColor = (0,255,255)
            print(flag)
            if(yawn_countdown and flag>=frame_thresh_3):
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsy after yawn", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                if not is_alert_playing:
                    play_alert()
                    is_alert_playing = True
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_2 and getFaceDirection(shape, size)<0):
                eyeContourColor = (255, 0, 0)
                cv2.putText(gray, "Drowsy (Body Posture)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                if not is_alert_playing:
                    play_alert()
                    is_alert_playing = True
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_1):
                eyeContourColor = (0, 0, 255)
                cv2.putText(gray, "Drowsy (Normal)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                if not is_alert_playing:
                    play_alert()
                    is_alert_playing = True
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
        elif(avgEAR>close_thresh and flag):
            print("Flag reseted to 0")
            is_alert_playing = False
            yawn_countdown=0
            map_flag=1
            flag=0

        if(map_counter>=redirect_threshold):
            # Only redirect if at least 5 minutes (300 seconds) have passed since the last redirect
            current_time = time.time()
            if current_time - last_redirect_time >= 300:
                map_flag = 1
                map_counter = 0
                play_break_alert()
                webbrowser.open("https://www.google.com/maps/search/hotels+or+motels+near+me")
                last_redirect_time = current_time
                print("Redirecting to find rest locations - next redirect available in 5 minutes")
            else:
                # Reset counter but don't redirect yet
                remaining_time = int(300 - (current_time - last_redirect_time))
                print(f"Too soon for another redirect. {remaining_time} seconds remaining.")
                map_counter = 0

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
    if(avgEAR>close_thresh):
        is_alert_playing = False
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()