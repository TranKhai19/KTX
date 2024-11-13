from flask import Flask, render_template, Response, jsonify
import face_recognition
import cv2
import imutils
import pickle
import time

app = Flask(__name__)

# Load encodings from file
with open('encodings.pickle', "rb") as file:
    data = pickle.load(file)

# Khởi tạo video stream (Webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Sử dụng DirectShow cho Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)


def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển frame từ BGR sang RGB và resize để tăng tốc độ xử lý
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(rgb, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # Detect và encoding các khuôn mặt trong frame
        boxes = face_recognition.face_locations(rgb, model="hog")  # có thể thay bằng "cnn" nếu muốn
        encodings = face_recognition.face_encodings(rgb, boxes)

        names = []  # Danh sách tên khuôn mặt đã nhận diện được

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)

        # Vẽ bounding boxes và tên trên frame
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode lại frame thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Truyền frame dưới dạng byte stream

@app.route('/')
def index():
    return render_template('recog.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
