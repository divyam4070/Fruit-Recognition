# from ultralytics import YOLO
# import cv2
# import numpy as np
# model = YOLO('FRUIT_MODEL.pt')

# result = model(source=0, show = True, conf= 0.2, save = False, imgsz=1024) #reduce imagesz if it lags
# from flask import Flask, render_template, Response
# from ultralytics import YOLO
# import cv2

# app = Flask(__name__)
# model = YOLO('FRUIT_MODEL.pt')  # Load your trained model

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # 0 = webcam
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             results = model(frame, conf=0.3, imgsz = 224)
#             annotated_frame = results[0].plot()
#             ret, buffer = cv2.imencode('.jpg', annotated_frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Load your trained YOLO model
model = YOLO('FRUIT_MODEL.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        img_data = base64.b64decode(data['image'].split(',')[1])

        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = np.array(img)

        # Predict
        results = model.predict(img, conf=0.6, imgsz=320, show = True, save = False)
        annotated_frame = results[0].plot()

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': 'data:image/jpeg;base64,' + encoded_image})

    except Exception as e:
        print("ERROR during prediction:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
