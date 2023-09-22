import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import mouse

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

##def draw_landmarks(image, results):
##    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
##    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
##    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
##    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
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

##cap = cv2.VideoCapture(0)
### Set mediapipe model 
##with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
##    while cap.isOpened():
##
##        # Read feed
##        ret, frame = cap.read()
##
##        # Make detections
##        image, results = mediapipe_detection(frame, holistic)
##        #print(results)
##        
##        # Draw landmarks
##        draw_styled_landmarks(image, results)
##
##        # Show to screen
##        cv2.imshow('OpenCV Feed', image)
##        
##        # Break gracefully
##        if cv2.waitKey(10) & 0xFF == ord('q'):
##            break
##    cap.release()
##    cv2.destroyAllWindows()

#ddraw_styled_landmarks(frame, results)
#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#left hand landmark
#print(len(results.left_hand_landmarks.landmark))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#result_test = extract_keypoints(results)
#np.save('0', result_test)
#np.load('0.npy')

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data') 

# Actions that we try to detect
actions = np.array(['none', 'fire', 'one_punch'])

# Thirty videos worth of data
no_sequences = 60

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 1

for action in actions: 
    #dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass



cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            
            while(True):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                cv2.putText(image, 'Next {} Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('e'):
                    break

                if mouse.is_pressed("right"):
                    break
            
            # Loop through video length aka sequence length
            for frame_num in range(1,sequence_length+1):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 1: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Next {} Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting {} Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()



##from sklearn.model_selection import train_test_split
##from tensorflow.keras.utils import to_categorical
##
##label_map = {label:num for num, label in enumerate(actions)}
##
##sequences, labels = [], []
##for action in actions:
##    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
##        window = []
##        for frame_num in range(1,sequence_length+1):
##            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
##            window.append(res)
##        sequences.append(window)
##        labels.append(label_map[action])
##
##X = np.array(sequences)
##y = to_categorical(labels).astype(int)
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
##
##
##
##from tensorflow.keras.models import Sequential
##from tensorflow.keras.layers import LSTM, Dense
##from tensorflow.keras.callbacks import TensorBoard
##
##log_dir = os.path.join('Logs')
##tb_callback = TensorBoard(log_dir=log_dir)
##
##model = Sequential()
##model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
##model.add(LSTM(128, return_sequences=True, activation='relu'))
##model.add(LSTM(64, return_sequences=False, activation='relu'))
##model.add(Dense(64, activation='relu'))
##model.add(Dense(32, activation='relu'))
##model.add(Dense(actions.shape[0], activation='softmax'))
##
##model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
##model.fit(X_train, y_train, epochs=150, callbacks=[tb_callback])
##
###model.summary()
##
##model.predict(X_test)
##
##model.save('action.h5')
