import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import mediapipe as mp
import numpy as np
import time
import pandas as pd

from numpy import dot
from numpy.linalg import norm
import math
import threading

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def distance(p1, p2):
    return math.dist((p1[0], p1[1]), (p2[0], p2[1])) 

def control_volume():
    if hand_shape == 'volume up' and 1 in data_df['select'].values:
        select_index = data_df.index[data_df['select'] == 1].to_list()
        if data_df.loc[select_index, 'status'].values[0] == 1:
            current_volume = data_df.loc[select_index, 'volume'].values[0]
            while True:
                if hand_shape != 'volume up' or result.multi_hand_landmarks is None:
                    break
                time.sleep(2)
                if current_volume < 10:
                    current_volume += 1
                data_df.loc[select_index, 'volume'] = current_volume          
    if hand_shape == 'volume down' and 1 in data_df['select'].values:
        select_index = data_df.index[data_df['select'] == 1].to_list()
        if data_df.loc[select_index, 'status'].values[0] == 1:
            current_volume = data_df.loc[select_index, 'volume'].values[0]
            while True:
                if hand_shape != 'volume down' or result.multi_hand_landmarks is None:
                    break
                time.sleep(2)
                if current_volume > 0:
                    current_volume -= 1
                data_df.loc[select_index, 'volume'] = current_volume
                
            
CONFIDENCE_THRESHOLD = 0.4
consine_threshold = 0.9
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLACK = (0,0,0)
class_list=['AC','lamp','laptop']

#yolo load model
model = YOLO('best.pt')
off_on = ['off','on']
max_num_hands = 2
gesture = {
    0: "fist",
    1: "point",  # "one"
    2: "two",
    3: "three",
    4: "four",
    5: "on/off",  # "five"
    6: "six",
    7: "rock",
    8: "spiderman",
    9: "yeah",
    10: "ok",
}
rps_gesture = {1: "point", 5: "on/off"}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt("gesture_train.csv", delimiter=",")
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cam Error')
        break

    frame = cv2.flip(frame,1)

    #object detection result
    detection = model.predict(source=[frame], save=False)[0]
    results = []

    for data in detection.boxes.data.tolist(): # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, class_list[label]+' '+str(round(confidence, 3)) + '%', (xmin, ymin), cv2.FONT_ITALIC, 1, WHITE, 2)
        cv2.line(frame,(xmin+int((xmax-xmin)/2),ymin+int((ymax-ymin)/2)),(xmin+int((xmax-xmin)/2),ymin+int((ymax-ymin)/2)), GREEN, 5) # 중심점
        results.append([xmin, ymin, xmax-xmin, ymax-ymin,xmin+int((xmax-xmin)/2),ymin+int((ymax-ymin)/2),confidence, class_list[label],0,0,0])
        data_df = pd.DataFrame(results, columns=['xmin','ymin','width','height','center_x','center_y','confidence','label','select','status','volume']) 
        data_df.to_csv('data_df.csv')
    
    cv2.imshow('frame', frame)
    # if time.sleep(10):
    #     break  
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# cv2.destroyAllWindows()
data_df = pd.read_csv('data_df.csv')
cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
while True:
    
    ret, frame = cap.read()
    if not ret:
        print('Cam Error')
        break
    
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)#색변환 rgb -> bgr
    for i in range(len(data_df)):
        cv2.rectangle(frame, (data_df['xmin'][i], data_df['ymin'][i]), (data_df['xmin'][i]+data_df['width'][i], data_df['ymin'][i]+data_df['height'][i]), RED, 2)
        cv2.putText(frame, text=data_df['label'][i]+' '+off_on[data_df['status'][i]]+' volume: '+str(data_df['volume'][i]),org=(data_df['xmin'][i],data_df['center_y'][i]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255,255,0),
            thickness=2,)
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            # 21개의 점의 좌표
            fingers = [0,0,0,0,0]
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] 
            if distance(joint[4],joint[9]) > distance(joint[3],joint[9]):
                fingers[0]=1
            for i in range(1, 5):  # 검지손가락 ~ 새끼손가락 순서로 확인
                if distance(joint[4 * (i + 1)], joint[0]) > distance(
                    joint[4 * (i + 1) - 1], joint[0]
                ):
                    fingers[i] = 1
            if fingers[0] == 1 and fingers[1:] == [0, 0, 0, 0] and joint[4][1]<joint[9][1]:  # 엄지손가락만 펴고 나머지 손가락이 모두 접힌 경우
                hand_shape = "volume up"
            elif fingers[0] == 1 and fingers[1:] == [0, 0, 0, 0] and joint[4][1]>joint[9][1]:
                hand_shape ="volume down"
            else:
                hand_shape=""      
            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(
                np.einsum(
                    "nt,nt->n",
                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                )
            )  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0]))
                index_finger = (int(res.landmark[8].x * frame.shape[1]), int(res.landmark[8].y * frame.shape[0]))
                cv2.putText(
                    frame,
                    text=rps_gesture[idx].upper(),
                    org=(org[0], org[1] + 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                )
            else:
                org = (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0]))
                index_finger = (int(res.landmark[8].x * frame.shape[1]), int(res.landmark[8].y * frame.shape[0]))
                cv2.putText(
                    frame,
                    text=hand_shape,
                    org=(org[0], org[1] + 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                )

            min_value = 10000
            
            #point 동작일 때
            if idx==1:
                cv2.line(frame, (org[0],org[1]),(index_finger[0],index_finger[1]), GREEN, 5)
                           
                #코사인 유사도가 1에 가장 가까운 index 추출
                for i in range(len(data_df)):
                    center_to_org =np.array([data_df['center_x'][i]-org[0],data_df['center_y'][i]-org[1]])
                    index_finger_to_org=np.array([index_finger[0]-org[0], index_finger[1]-org[1]])                    
                    cosine_similarity=cos_sim(center_to_org,index_finger_to_org)
                    if cosine_similarity > consine_threshold:
                        if min_value > 1-cosine_similarity: 
                            min_value = 1-cosine_similarity
                            min_index = i
                            data_df.loc[min_index,'select']=1
                            data_df.loc[data_df.index != min_index, 'select'] = 0
                    else:
                        data_df.loc[i,'select']=0                

            if idx==5 and 1 in data_df['select'].values:    
                select_index = data_df.index[data_df['select']==1].to_list()
                if (data_df.loc[select_index,'status']==0).any():
                    data_df.loc[select_index,'status']=1
                elif (data_df.loc[select_index,'status']==1).any():
                    data_df.loc[select_index,'status']=0
                data_df.loc[select_index,'select']=0
            thread = threading.Thread(target=control_volume)
            thread.daemon =True
            thread.start()

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
    if 1 in data_df['select'].values:
        select_index = data_df.index[data_df['select']==1].to_list()
        cv2.line(frame,(org[0],org[1]),(data_df['center_x'][select_index[0]],data_df['center_y'][select_index[0]]),GREEN,5)
        cv2.rectangle(frame, (data_df['xmin'][select_index[0]], data_df['ymin'][select_index[0]]), (data_df['xmin'][select_index[0]]+data_df['width'][select_index[0]], data_df['ymin'][select_index[0]]+data_df['height'][select_index[0]]), GREEN, 2)

    # 손이 감지되지 않을 때 선택 없애기
    if result.multi_hand_landmarks is None:
        data_df['select']=0

    data_df.to_csv('data_df.csv')
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
