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

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

CONFIDENCE_THRESHOLD = 0.4
consine_threshold = 0.9
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
class_list=['AC','Window-blind','lamp','laptop']
#yolo load model
model = YOLO('best.pt')
off_on = ['off','on']
#tracker = DeepSort(max_age=50)
motion_duration = 3
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
rps_gesture = {0:'off',1: "point", 5: "on"}

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
    start = datetime.datetime.now()

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
        results.append([xmin, ymin, xmax-xmin, ymax-ymin,xmin+int((xmax-xmin)/2),ymin+int((ymax-ymin)/2),confidence, class_list[label],0,0])
        data_df = pd.DataFrame(results, columns=['xmin','ymin','width','height','center_x','center_y','confidence','label','select','status']) 
        data_df.to_csv('data_df.csv')
    #tracks = tracker.update_tracks(results, frame=frame)

    # for track in tracks:
    #     if not track.is_confirmed():
    #         continue

    #     track_id = track.track_id
    #     ltrb = track.to_ltrb()

    #     xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
    #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
    #     cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    # end = datetime.datetime.now()

    # total = (end - start).total_seconds()
    # # print(f'Time to process 1 frame: {total * 1000:.0f} milliseconds')
    # fps = f'FPS: {1 / total:.2f}'
    # cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
#select_list=[]
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
    if result.multi_hand_landmarks is not None:
        #rps_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            # 21개의 점의 좌표
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] 

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

                #rps_result.append({"rps": rps_gesture[idx], "org": org})
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
                            data_df['select'][min_index]=1
                            data_df.loc[data_df.index != min_index, 'select'] = 0
                    else:
                        data_df['select'][i]=0
                                  
            # 선택된 객체가 있고 on/off 하는 동작일 때    
            if idx==0 and 1 in data_df['select'].values:
                select_index = data_df.index[data_df['select']==1].to_list()
                if data_df['status'][select_index[0]]==1:
                    data_df['status'][select_index[0]]=0
            if idx==5 and 1 in data_df['select'].values:    
                select_index = data_df.index[data_df['select']==1].to_list()
                if data_df['status'][select_index[0]]==0:
                    data_df['status'][select_index[0]]=1
                    
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    # 선택된 객체의 상태 보여주기        
    if 1 in data_df['select'].values:
        select_index = data_df.index[data_df['select']==1].to_list()
        cv2.line(frame,(org[0],org[1]),(data_df['center_x'][select_index[0]],data_df['center_y'][select_index[0]]),GREEN,5)
        cv2.rectangle(frame, (data_df['xmin'][select_index[0]], data_df['ymin'][select_index[0]]), (data_df['xmin'][select_index[0]]+data_df['width'][select_index[0]], data_df['ymin'][select_index[0]]+data_df['height'][select_index[0]]), GREEN, 2)
        cv2.putText(frame, text=data_df['label'][select_index[0]]+' '+off_on[data_df['status'][select_index[0]]],org=(data_df['center_x'][select_index[0]],data_df['center_y'][select_index[0]]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=GREEN,
                thickness=2,)
        
    # 손이 감지되지 않을 때 선택 없애기
    if result.multi_hand_landmarks is None:
        data_df['select']=0

    data_df.to_csv('data_df.csv')
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()