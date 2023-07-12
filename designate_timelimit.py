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
    return dot(A, B) / (norm(A) * norm(B))


CONFIDENCE_THRESHOLD = 0.4
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
class_list = ["AC", "Window-blind", "lamp", "laptop"]
# yolo load model
model = YOLO("best.pt")

max_num_hands = 1  # 인식 손 개수
gesture = {
    0: "fist",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "rock",
    8: "spiderman",
    9: "yeah",
    10: "ok",
}
rps_gesture = {
    1: "point",
    0: "off",
    5: "on",
    8: "reset",
    10: "control state",
}  # 검지손가락으로 가리키기 / 주먹 / 보자기 / 스파이더맨 / ok표시(엄지[4], 검지[8] 사이 거리로 조절)

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

    ret, frame = cap.read()  # 프레임 하나씩 읽음
    if not ret:  # 읽어오기 실패시
        print("Cam Error")
        break

    frame = cv2.flip(frame, 1)

    # object detection result
    detection = model.predict(source=[frame], save=False)[0]
    results = []

    for (
        data
    ) in (
        detection.boxes.data.tolist()
    ):  # data : [xmin, ymin, xmax, ymax, confidence_score, class_id, status]  # frame마다 데이터 갱신 q:종료
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])
        if label == 0:  # AC green
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        elif label == 1:  # Window-blind red
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        elif label == 2:  # lamp blue
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        else:  # laptop yellow
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(
            frame,
            class_list[label] + " " + str(round(confidence, 3)) + "%",
            (xmin, ymin),
            cv2.FONT_ITALIC,
            1,
            (255, 255, 255),
            2,
        )
        # 중심점
        cv2.line(
            frame,
            (xmin + int((xmax - xmin) / 2), ymin + int((ymax - ymin) / 2)),
            (xmin + int((xmax - xmin) / 2), ymin + int((ymax - ymin) / 2)),
            GREEN,
            5,
        )
        status = 0  # 초기 상태 0(off)로 설정
        results.append(
            [
                xmin,
                ymin,
                xmax - xmin,  # width
                ymax - ymin,  # height
                xmin + int((xmax - xmin) / 2),  # center x
                ymin + int((ymax - ymin) / 2),  # center y
                confidence,
                class_list[label],
                status,  # status
            ]
        )
        data_df = pd.DataFrame(
            results,
            columns=[
                "xmin",
                "ymin",
                "width",
                "height",
                "center_x",
                "center_y",
                "confidence",
                "label",
                "status",
            ],
        )
        data_df.to_csv("data_df.csv", index=False)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

ch_status = 99  # class name
pointing = True
ori_idx = 99
start_time = 0
while True:
    start = datetime.datetime.now()

    ret, frame = cap.read()
    if not ret:
        print("Cam Error")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 색변환 rgb -> bgr

    if result.multi_hand_landmarks is not None:
        rps_result = []
        data_df = pd.read_csv("data_df.csv")
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

                rps_result.append({"rps": rps_gesture[idx], "org": org})

            # 물체 지정
            min_value = 10000

            if pointing and idx == 1:  # 손동쟉 pointing 일 때 출력
                # org:손목점 x,y  index_finger:검지손가락 x,y
                cv2.line(frame, (org[0], org[1]), (index_finger[0], index_finger[1]), GREEN, 5)

                # 기울기 차이가 가장 적은 index 추출
                def fix_idx(data_df, org, index_finger, min_value, ch_status, ori_idx, start_time):
                    for i in range(len(data_df)):
                        center_to_org = np.array([data_df["center_x"][i] - org[0], data_df["center_y"][i] - org[1]])
                        index_finger_to_org = np.array([index_finger[0] - org[0], index_finger[1] - org[1]])
                        cos_val = cos_sim(center_to_org, index_finger_to_org)

                        if min_value > 1 - cos_val:
                            min_value = 1 - cos_val
                            min_index = i
                            ch_status = data_df["label"][i]  # 지정한 물체 class index 저장
                    cv2.line(
                        frame,
                        (org[0], org[1]),
                        (data_df["center_x"][min_index], data_df["center_y"][min_index]),
                        GREEN,
                        5,
                    )  # cos값 가장 작은 물체와 손목점 사이 직선 표시
                    if ori_idx == 99:
                        ori_idx = min_index
                        start_time = datetime.datetime.now()
                    return min_index, ch_status, ori_idx, start_time

                fidx, ch_status, ori_idx, start_time = fix_idx(
                    data_df, org, index_finger, min_value, ch_status, ori_idx, start_time
                )
                if (
                    start_time != 0 and ori_idx == fidx and (datetime.datetime.now() - start_time).seconds >= 3
                ):  # 물체를 지정한 채로 3초가 지나면 물체 고정 및 포인팅 동작 인식 X
                    pointing = False
                    print(f"지정한 물체: {ch_status}")
                elif ori_idx != fidx:
                    ori_idx = fidx
                    start_time = datetime.datetime.now()
            if not pointing:
                if idx == 0 and ch_status != "Window-blind":  # off
                    for i in range(len(data_df)):
                        if data_df["label"][i] == ch_status:  # 지정한 클래스인 경우 상태정보 변경
                            data_df.loc[i, "status"] = 0
                            # break  # 같은 클래스의 경우 상태정보 일괄 변경
                    cv2.putText(
                        frame,
                        "turn off",
                        (data_df["xmin"][i], data_df["ymin"][i]),
                        cv2.FONT_ITALIC,
                        1,
                        (255, 255, 255),
                        2,
                    )
                elif idx == 5 and ch_status != "Window-blind":  # on
                    for i in range(len(data_df)):
                        if data_df["label"][i] == ch_status:
                            data_df.loc[i, "status"] = 5
                            # break
                    cv2.putText(
                        frame,
                        "turn on",
                        (data_df["xmin"][i], data_df["ymin"][i]),
                        cv2.FONT_ITALIC,
                        1,
                        (255, 255, 255),
                        2,
                    )
                elif idx == 10:  # change status
                    thumb = res.landmark[4]
                    index = res.landmark[8]

                    diff = abs(index.x - thumb.x) * 1000  # 검지, 엄지 사이 거리

                    status_value = min(int(diff), 100)  # 최대
                    for i in range(len(data_df)):
                        if data_df["label"][i] == ch_status:
                            data_df.loc[i, "status"] = 10  # status_value로 상태 저장
                            # break
                    cv2.putText(
                        frame,
                        text="control status: %d" % status_value,
                        org=(data_df["xmin"][i], data_df["ymin"][i]),
                        fontFace=cv2.FONT_ITALIC,  # FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                    )
                elif idx == 8:  # reset designated object
                    pointing = True
                    ori_idx = 99

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        data_df.to_csv("data_df.csv")
        print("data has been updated")
        break

cap.release()
cv2.destroyAllWindows()
