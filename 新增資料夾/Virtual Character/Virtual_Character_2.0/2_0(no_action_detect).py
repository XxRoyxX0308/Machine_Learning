from filterpy.kalman import KalmanFilter

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import socket

kf=[[]*3 for i in range(17)]
for i in range(0,17):
    for j in range(0,3):
        kf[i].append(KalmanFilter(dim_x=1,dim_z=1))
        kf[i][j].x=np.array([1])
        kf[i][j].F=np.array([1.])
        kf[i][j].H=np.array([1.])
        kf[i][j].R*=0.1**2
        kf[i][j].P*=500
        kf[i][j].Q=1e-3

HOST = '127.0.12.12'
PORT = 1225

poselist=[11,13,15,10,12,14,16,23,25,27,24,26,28,0,9,10]

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

##def draw_landmarks(image, results):
##    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
##    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
##    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
##    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
##    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
##                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
##                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
##                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results_RollPitchYaw = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        if results_RollPitchYaw.multi_face_landmarks:
            for face_landmarks in results_RollPitchYaw.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360


        
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        #print(results.pose_landmarks.landmark[16].x)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        
        message=""
        try:
            for i in range(0,16):
                kf[i][0].update(results.pose_landmarks.landmark[poselist[i]].x,0.1**2,np.array([1]))
                kf[i][0].predict()
                results.pose_landmarks.landmark[poselist[i]].x=float(kf[i][0].x)

                kf[i][1].update(results.pose_landmarks.landmark[poselist[i]].y,0.1**2,np.array([1]))
                kf[i][1].predict()
                results.pose_landmarks.landmark[poselist[i]].y=float(kf[i][1].x)

                kf[i][2].update(results.pose_landmarks.landmark[poselist[i]].z,0.1**2,np.array([1]))
                kf[i][2].predict()
                results.pose_landmarks.landmark[poselist[i]].z=float(kf[i][2].x)

            kf[16][0].update(x,0.1**2,np.array([1]))
            kf[16][0].predict()
            x=float(kf[16][0].x)-6

            kf[16][1].update(y,0.1**2,np.array([1]))
            kf[16][1].predict()
            y=float(kf[16][1].x)

            kf[16][2].update(z,0.1**2,np.array([1]))
            kf[16][2].predict()
            z=float(kf[16][2].x)


            
            #LeftArm
            message+=str(round(results.pose_landmarks.landmark[11].x,5))+" "+str(round(results.pose_landmarks.landmark[11].y,5))+" "+str(round(results.pose_landmarks.landmark[11].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[13].x,5))+" "+str(round(results.pose_landmarks.landmark[13].y,5))+" "+str(round(results.pose_landmarks.landmark[13].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[15].x,5))+" "+str(round(results.pose_landmarks.landmark[15].y,5))+" "+str(round(results.pose_landmarks.landmark[15].z,5))+" "

            #Neck ?
            message+=str(round(results.pose_landmarks.landmark[10].x,5))+" "+str(round(results.pose_landmarks.landmark[10].y,5))+" "+str(round(results.pose_landmarks.landmark[10].z,5))+" "
            
            #RightArm
            message+=str(round(results.pose_landmarks.landmark[12].x,5))+" "+str(round(results.pose_landmarks.landmark[12].y,5))+" "+str(round(results.pose_landmarks.landmark[12].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[14].x,5))+" "+str(round(results.pose_landmarks.landmark[14].y,5))+" "+str(round(results.pose_landmarks.landmark[14].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[16].x,5))+" "+str(round(results.pose_landmarks.landmark[16].y,5))+" "+str(round(results.pose_landmarks.landmark[16].z,5))+" "



            #LeftThigh
            message+=str(round(results.pose_landmarks.landmark[23].x,5))+" "+str(round(results.pose_landmarks.landmark[23].y,5))+" "+str(round(results.pose_landmarks.landmark[23].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[25].x,5))+" "+str(round(results.pose_landmarks.landmark[25].y,5))+" "+str(round(results.pose_landmarks.landmark[25].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[27].x,5))+" "+str(round(results.pose_landmarks.landmark[27].y,5))+" "+str(round(results.pose_landmarks.landmark[27].z,5))+" "

            #RughtThigh
            message+=str(round(results.pose_landmarks.landmark[24].x,5))+" "+str(round(results.pose_landmarks.landmark[24].y,5))+" "+str(round(results.pose_landmarks.landmark[24].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[26].x,5))+" "+str(round(results.pose_landmarks.landmark[26].y,5))+" "+str(round(results.pose_landmarks.landmark[26].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[28].x,5))+" "+str(round(results.pose_landmarks.landmark[28].y,5))+" "+str(round(results.pose_landmarks.landmark[28].z,5))+" "

            #Nose
            message+=str(round(results.pose_landmarks.landmark[0].x,5))+" "+str(round(results.pose_landmarks.landmark[0].y,5))+" "+str(round(results.pose_landmarks.landmark[0].z,5))+" " 

            #mouth
            message+=str(round(results.pose_landmarks.landmark[9].x,5))+" "+str(round(results.pose_landmarks.landmark[9].y,5))+" "+str(round(results.pose_landmarks.landmark[9].z,5))+" "
            message+=str(round(results.pose_landmarks.landmark[10].x,5))+" "+str(round(results.pose_landmarks.landmark[10].y,5))+" "+str(round(results.pose_landmarks.landmark[10].z,5))+" "

            #Neck RollPitchYaw
            message+=str(round(x,5))+" "+str(round(y,5))+" "+str(round(z,5))+" "
        except:
            for i in range(0,16*3):
                message+="0 "

##        try:
##            #LeftHand
##            message+=str(round(results.left_hand_landmarks.landmark[9].x,5))+" "+str(round(results.left_hand_landmarks.landmark[9].y,5))+" "+str(round(results.left_hand_landmarks.landmark[9].z,5))+" "
##            message+=str(round(results.left_hand_landmarks.landmark[0].x,5))+" "+str(round(results.left_hand_landmarks.landmark[0].y,5))+" "+str(round(results.left_hand_landmarks.landmark[0].z,5))+" "
##
##            #RightHand
##            message+=str(round(results.right_hand_landmarks.landmark[9].x,5))+" "+str(round(results.right_hand_landmarks.landmark[9].y,5))+" "+str(round(results.right_hand_landmarks.landmark[9].z,5))+" "
##            message+=str(round(results.right_hand_landmarks.landmark[0].x,5))+" "+str(round(results.right_hand_landmarks.landmark[0].y,5))+" "+str(round(results.right_hand_landmarks.landmark[0].z,5))+" "
##        except:
##            for i in range(0,4*3):
##                message+="0 "

        try:
            s.send(message.encode())
        except:
            pass
        
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

##draw_landmarks(frame, results)
##plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
