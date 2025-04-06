# from ultralytics import YOLO
# import cv2
# import numpy as np
# model = YOLO('FRUIT_MODEL.pt')

# result = model(source=0, show = True, conf= 0.2, save = False, imgsz=1024) #reduce imagesz if it lags
from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO('FRUIT_MODEL.pt')  # Load your trained model

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame, conf=0.3, imgsz = 224)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
