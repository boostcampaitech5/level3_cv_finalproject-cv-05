from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

app = FastAPI()

class_list = ["AC", "lamp", "laptop"]

CONFIDENCE_THRESHOLD = 0.7
# yolo load model
model = YOLO("best_latest.pt")
# 118.67.143.219

@app.websocket("/ws")
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
            if label == 0:  # AC green
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            elif label == 1:  # Window-blind red
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            elif label == 2:  # lamp blue
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            else:  # laptop yellow
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

            # 화면상 label 표시
            cv2.putText(
                frame,
                class_list[label] + " " + str(round(confidence, 3)) + "%",
                (xmin, ymin),
                cv2.FONT_ITALIC,
                1,
                (255, 255, 255),
                1,
            )
            # 중심점
            cv2.line(
                frame,
                (xmin + int((xmax - xmin) / 2), ymin + int((ymax - ymin) / 2)),
                (xmin + int((xmax - xmin) / 2), ymin + int((ymax - ymin) / 2)),
                (0, 255, 0),
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

        bbox_data = data_df.to_dict("records")

        await websocket.send_json(bbox_data)  # WebSocket을 통해 JSON 데이터를 클라이언트로 전송하는 비동기 함수


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
        <canvas id="canvas" width="640" height="480"></canvas>

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.15.0"></script>
        <script>
        var video = document.querySelector("#video");
        var canvas = document.querySelector("#canvas");
        var socket;
        var isWebSocketOpen = false;

        function startWebSocket() {
            var constraints = { video: true };

            navigator.mediaDevices.getUserMedia(constraints)
            .then(function (stream) {
                video.srcObject = stream;
                socket = new WebSocket("ws://localhost:30006/ws");
                socket.addEventListener("open", function (event) {
                isWebSocketOpen = true;
                document.getElementById("startButton").disabled = true;
                document.getElementById("stopButton").disabled = false;
                setInterval(sendImage, 200); // Send image data every second(1000ms=1sec) -> 빈도수를 늘리려면 시간을 짧게 설정(1000->200)
                });
                socket.addEventListener("message", function (event) {
                console.log("Message from server: ", event.data);
                var bboxData = JSON.parse(event.data);
                displayBoundingBoxes(bboxData);
                });
            })
            .catch(function (err) {
                console.log("An error occurred: " + err);
            });
        }

        function stopWebSocket() {
            if (isWebSocketOpen) {
            socket.close();
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

        function sendImage() {
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            var imageData = canvas.toDataURL("image/png");
            socket.send(imageData);
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
