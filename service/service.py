import asyncio
import base64
import cv2
import json
import math
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import subprocess
import threading
import time

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from google.protobuf.json_format import MessageToJson
from numpy.linalg import norm
from ultralytics import YOLO

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

app = FastAPI()

class_list = ["AC", "lamp", "laptop"]

CONFIDENCE_THRESHOLD = 0.7
# yolo load model
model = YOLO("best.pt")

@app.websocket("/bd")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket connection established")

    while True:
        data = await websocket.receive_text()
        # We are receiving the base64 image here, we will decode and process the image.
        base64_img = data.split(",")[1]
        decoded_img = base64.b64decode(base64_img)

        # Convert decoded image to numpy array
        nparr = np.frombuffer(decoded_img, np.uint8)

        # Decode image as BGR
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image with object detection model using `frame`
        frame = cv2.flip(frame, 1)

        # object detection result
        detection = model.predict(source=[frame], save=False)[0]
        results = []

        for data in detection.boxes.data.tolist():
            # data : [xmin, ymin, xmax, ymax, confidence_score, class_id, status]
            confidence = float(data[4])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])
            results.append([xmin, ymin, xmax-xmin, ymax-ymin,xmin+int((xmax-xmin)/2),ymin+int((ymax-ymin)/2),confidence, class_list[label],0,0,0])
        data_df = pd.DataFrame(results, columns=['xmin','ymin','width','height','center_x','center_y','confidence','label','select','status','volume']) 
        
        data_df.to_csv("data_df.csv", index=False)

        bbox_data = data_df.to_dict("records")

        await websocket.send_json(bbox_data)  # WebSocket을 통해 JSON 데이터를 클라이언트로 전송하는 비동기 함수


@app.websocket("/hd")
async def video_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket connection established")

    def cos_sim(A, B):
        return np.dot(A, B)/(norm(A)*norm(B))

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
                    time.sleep(1)
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
                    time.sleep(1)
                    if current_volume > 0:
                        current_volume -= 1
                    data_df.loc[select_index, 'volume'] = current_volume

    consine_threshold = 0.9
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    off_on = ['off','on']
    max_num_hands = 1
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

    data_file = './data_df.csv'
    if os.path.exists(data_file):
        data_df = pd.read_csv('data_df.csv')

    while True:
        data = await websocket.receive_text()
        base64_img = data.split(",")[1]
        decoded_img = base64.b64decode(base64_img)

        # Convert decoded image to numpy array
        nparr = np.frombuffer(decoded_img, np.uint8)

        # Decode image as BGR
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)#색변환 rgb -> bgr
        # bbox 표시하기
        # for i in range(len(data_df)):
        #     cv2.rectangle(frame, (data_df['xmin'][i], data_df['ymin'][i]), (data_df['xmin'][i]+data_df['width'][i], data_df['ymin'][i]+data_df['height'][i]), RED, 2)
        #     cv2.putText(frame, text=data_df['label'][i]+' '+off_on[data_df['status'][i]]+' volume: '+str(data_df['volume'][i]),org=(data_df['xmin'][i],data_df['center_y'][i]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=1,
        #         color=(255,255,0),
        #         thickness=3,)
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

                else:
                    org = (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0]))
                    index_finger = (int(res.landmark[8].x * frame.shape[1]), int(res.landmark[8].y * frame.shape[0]))

                min_value = 10000
                
                #point 동작일 때
                if idx==1:
                    # cv2.line(frame, (org[0],org[1]),(index_finger[0],index_finger[1]), GREEN, 5)
                            
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
            # cv2.line(frame,(org[0],org[1]),(data_df['center_x'][select_index[0]],data_df['center_y'][select_index[0]]),GREEN,5)
            # cv2.rectangle(frame, (data_df['xmin'][select_index[0]], data_df['ymin'][select_index[0]]), (data_df['xmin'][select_index[0]]+data_df['width'][select_index[0]], data_df['ymin'][select_index[0]]+data_df['height'][select_index[0]]), GREEN, 2)

        # 손이 감지되지 않을 때 선택 없애기
        if result.multi_hand_landmarks is None:
            data_df['select']=0

        data_df.to_csv('data_df.csv')
        

        if result.multi_hand_landmarks is not None:    
            hand_data = result.multi_hand_landmarks[0]

            # 'landmark_list'는 NormalizedLandmarkList 객체라고 가정합니다.

            json_str = MessageToJson(hand_data)
            hand_dict = json.loads(json_str)
            landmark_data = hand_dict["landmark"]
            bbox_data = data_df.to_dict("records")
            
            # 두 데이터를 딕셔너리로 묶기
            data_to_send = [landmark_data, bbox_data, idx]

            
            await websocket.send_json(data_to_send)
        else:
            bbox_data = data_df.to_dict("records")
            
            # 두 데이터를 딕셔너리로 묶기
            data_to_send = [bbox_data]
            await websocket.send_json(data_to_send)
            
        

@app.websocket("/ud")
async def text_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket connection established")
    while True:
        data = await websocket.receive_text()
        if isinstance(data, str) and data.startswith("model"):
            data = data.split(",")[1]
            class_name=data.split(" ")[0]
            model_name=data.split(" ")[1]
            print(class_name,model_name)
            subprocess.run(["python","data_collection.py","--class_name",class_name,"--model_name",model_name])
            subprocess.run(["python","mk_new_dataset.py"])
            subprocess.run("/opt/ml/final_project/update_train.sh",shell=True)
            config_path = 'config.yaml'
            config = load_config(config_path)
            model_version = config['model_version']
            if model_version==1:
                model_version=""
            else:
                model_version=str(model_version)
            print(model_version)
            await websocket.send_text("Training started.")

@app.get("/", response_class=HTMLResponse)
async def read_items():
    html_content = """
    <html>
    <head>
        <title>Object Detection with FastAPI</title>
    </head>
    <body>
        <h1>Object Detection with FastAPI</h1>
        <video id="video" width="640" height="480" autoplay></video>
        <button id="startButton" onclick="startWebSocket()">Start WebSocket</button>
        <button id="stopButton" onclick="stopWebSocket()" disabled>Stop WebSocket</button>
        <button id="handstart" onclick="startHandRecognition()">Hand Start</button>
        <button id="handstop" onclick="stopHandWebSocket()" disabled>Hand Stop</button>
        <canvas id="canvas" width="640" height="480"></canvas>
        <input type="text" id="messageText" autocomplete="off"/>
        <button onclick="sendText()">Send Text</button>
        
        
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.15.0"></script>
        <script>
        var video = document.querySelector("#video");
        var canvas = document.querySelector("#canvas");
        var socket;
        var handSocket;
        var isWebSocketOpen = false;
        var isHandWebSocketOpen = false;

        
        function startWebSocket() {
            var constraints = { video: true };

            navigator.mediaDevices.getUserMedia(constraints)
            .then(function (stream) {
                video.srcObject = stream;
                socket = new WebSocket("ws://118.67.143.219:30006/bd");
                socket.addEventListener("open", function (event) {
                    isWebSocketOpen = true;
                    document.getElementById("startButton").disabled = true;
                    document.getElementById("stopButton").disabled = false;
                    setInterval(sendImage, 500);
                });
                socket.addEventListener("message", function (event) {
                    console.log("Message from server: ", event.data);
                    try {
                        var bboxData = JSON.parse(event.data);
                        displayBoundingBoxes(bboxData);
                    } catch (error) {
                        console.error("Error parsing JSON:", error);
                    }
                });
            })
            .catch(function (err) {
                console.log("An error occurred: " + err);
            });
        }

        function stopWebSocket() {
            if (isWebSocketOpen) {
                socket.close();
                isWebSocketOpen = false;
                document.getElementById("startButton").disabled = false;
                document.getElementById("stopButton").disabled = true;
            }
        }
        

        function displayBoundingBoxes(bboxData) {
            var context = canvas.getContext("2d");

            for (var i = 0; i < bboxData.length; i++) {
                var bbox = bboxData[i];
                var xmin = bbox.xmin;
                var ymin = bbox.ymin;
                var width = bbox.width;
                var height = bbox.height;
                var xmax = xmin + width;
                var ymax = ymin + height;
                var label = bbox.label;

                // Draw bounding box
                context.beginPath();
                context.lineWidth = 2;
                context.strokeStyle = "red";
                context.rect(canvas.width - xmin - width, ymin, width, height);
                context.stroke();

                // Display class label
                context.font = "10px Arial";
                context.fillStyle = "red";
                context.fillText(label, canvas.width - xmin - width, ymin - 10);
            }
        }
            
        function displayBBox(bboxData) {
            var context = canvas.getContext("2d");

            for (var i = 0; i < bboxData.length; i++) {
                var bbox = bboxData[i];
                var xmin = bbox.xmin;
                var ymin = bbox.ymin;
                var width = bbox.width;
                var height = bbox.height;
                var xmax = xmin + width;
                var ymax = ymin + height;
                var label = bbox.label;
                var select = bbox.select;
                var status = bbox.status;
                var volume = bbox.volume;
                
                // Draw bounding box
                var boxColor = select === 1 ? "green" : "red";
                context.beginPath();
                context.lineWidth = 2;
                context.strokeStyle = boxColor;
                context.rect(canvas.width - xmin - width, ymin, width, height);
                context.stroke();

                // Display class label
                context.font = "10px Arial";
                context.fillStyle = boxColor;
                context.fillText(label + " | status: " + (status === 0 ? "off" : "on") + " | volume: " + volume, canvas.width - xmin - width, ymin - 10);

            }
        }

        function sendImage() {
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            var imageData = canvas.toDataURL("image/png");
            if (isWebSocketOpen) {
                socket.send(imageData);
            }
        }
        
        function startHandRecognition() {
            var constraints = { video: true };

            navigator.mediaDevices.getUserMedia(constraints)
            .then(function (stream) {
                video.srcObject = stream;
                handSocket = new WebSocket("ws://118.67.143.219:30006/hd"); // 프로토콜 수정
                handSocket.addEventListener("open", function (event) {
                    isHandWebSocketOpen = true;
                    document.getElementById("handstart").disabled = true;
                    document.getElementById("handstop").disabled = false;
                    setInterval(sendHandImage, 500);
                });
                handSocket.addEventListener("message", function (event) {
                    console.log("hand gesture result: ", event.data);
                    try {
                        var data = JSON.parse(event.data); // 전체 데이터를 먼저 파싱
                        console.log("data result: ", data);

                        if (data.length===3){
                            var handData = data[0]; 
                            var bboxData = data[1];
                            var idx = data[2]
                            displayBBoxLines(bboxData, handData, idx);
                        }
                        else{
                            var bboxData = data[0];
                            displayBBox(bboxData);
                        }
                    } catch (error) {
                        console.error("Error parsing JSON:", error);
                    }
                });
            })
            .catch(function (err) {
                console.log("에러 발생: " + err);
            });
        }   
            
        function stopHandWebSocket() {
            if (isHandWebSocketOpen) { // 변수명 수정
                handSocket.close(); // 변수명 수정
                isHandWebSocketOpen = false;
                document.getElementById("handstart").disabled = false;
                document.getElementById("handstop").disabled = true;
            }
        }
        
        function displayFingerLandmarks(handData) {
            var context = canvas.getContext("2d");

            for (var i = 0; i < 21; i++) {
                var hand = handData[i];
                var x = hand.x;
                var y = hand.y;
                
                // Draw landmark
                context.beginPath();
                context.fillStyle = 'blue';
                context.arc((1-x)*canvas.width, y*canvas.height, 3, 0, 2 * Math.PI);
                context.fill();
            }
        }

        function displayBBoxLines(bboxData, handData, idx) {
            var context = canvas.getContext("2d");

            for (var i = 0; i < bboxData.length; i++) {
                var bbox = bboxData[i];
                var xmin = bbox.xmin;
                var ymin = bbox.ymin;
                var width = bbox.width;
                var height = bbox.height;
                var xmax = xmin + width;
                var ymax = ymin + height;
                var label = bbox.label;
                var select = bbox.select;
                var status = bbox.status;
                var volume = bbox.volume;
                
                // Draw bounding box
                var boxColor = select === 1 ? "green" : "red";
                context.beginPath();
                context.lineWidth = 2;
                context.strokeStyle = boxColor;
                context.rect(canvas.width - xmin - width, ymin, width, height);
                context.stroke();

                // Display class label
                context.font = "10px Arial";
                context.fillStyle = boxColor;
                context.fillText(label + " | status: " + (status === 0 ? "off" : "on") + " | volume: " + volume, canvas.width - xmin - width, ymin - 10);


                // 손동작이 포인팅일때 지정된 물체에 대해서만 직선이 그어지도록 조건문 추가
                if (idx===1){
                    if (select===1){
                        // Calculate bbox center
                        var bboxCenterX = (canvas.width - xmin - width / 2);
                        var bboxCenterY = (ymin + height / 2);

                        // Connect bbox center to hand landmarks
                        var handX = handData[8].x; // 손목의 x 좌표
                        var handY = handData[8].y; // 손목의 y 좌표

                        // Draw line connecting bbox center and hand landmark (손목)
                        context.beginPath();
                        context.strokeStyle = "green";
                        context.moveTo(bboxCenterX, bboxCenterY);
                        context.lineTo((1 - handX) * canvas.width, handY * canvas.height);
                        context.stroke();
                    }
                }
                // Draw landmarks (파란색)
                for (var j = 0; j < 21; j++) {
                    var hand = handData[j];
                    var x = hand.x;
                    var y = hand.y;

                    // Draw landmark
                    context.beginPath();
                    context.fillStyle = 'blue';
                    context.arc((1 - x) * canvas.width, y * canvas.height, 2, 0, 2 * Math.PI);
                    context.fill();
                }
            }
        }
        
        function sendHandImage() {
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            var imageData = canvas.toDataURL("image/png");
            if (isHandWebSocketOpen) {
                handSocket.send(imageData);
            }
        }
        
        function sendText() {
            var text = document.getElementById("messageText").value;
            if (text.trim()) {  // 텍스트가 비어있지 않은 경우에만 실행
                var data = "model," + text;
                nw.send(data);
            }
        }
        
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


# --------------------------------------------------------------------------------------------
