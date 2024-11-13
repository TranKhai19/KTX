from flask import Flask, Response, render_template, jsonify
import cv2
import pickle
import face_recognition
import imutils
from datetime import datetime
import os
import csv

# Initialize Flask app
app = Flask(__name__)
with open('encodings.pickle', "rb") as file:
    data = pickle.load(file)
# Load the video capture from your camera or video file
video_capture = cv2.VideoCapture(0)  # Change '0' to your video source if needed
recognized_name = "Unknown"
attendance_time = "--:--:--"
csv_file = 'attendance_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time"]) 
def is_name_in_csv(name):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == name:  # Kiểm tra cột "Name" (cột đầu tiên)
                return True
    return False

import datetime

def gen_frames():
    global recognized_name, attendance_time  # Khai báo biến global
    recognized_name = "Unknown"  # Khởi tạo tên mặc định
    attendance_time = "--:--:--"  # Khởi tạo thời gian mặc định

    while True:
        ret, frame = video_capture.read()
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

        # Cập nhật recognized_name và attendance_time nếu nhận diện được khuôn mặt mới
        if names:
            if recognized_name != names[0]:  # Nếu nhận diện khuôn mặt khác so với lần trước
                recognized_name = names[0]
                attendance_time = datetime.datetime.now().strftime("%H:%M:%S")  # Lấy thời gian hiện tại

                if not is_name_in_csv(recognized_name):
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([recognized_name, attendance_time])
        else:
            recognized_name = "Unknown"
            attendance_time = "--:--:--"  # Đặt lại thời gian nếu không nhận diện được

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
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/screen')
def screen():
    return render_template('screen.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_data')
def recognition_data():
    global recognized_name, attendance_time
    print("Recognized Name:", recognized_name)
    print("Attendance Time:", attendance_time)
    return jsonify({
        'name': recognized_name,
        'time': attendance_time
    })

# Home route for testing
@app.route('/recognition')
def recognition():
    return '''
    <html>
        <body>
            <h1>Face Recognition Video Stream</h1>
            <img src="/video_feed" width="640" height="480" />

            <div class="info-grid">
                <div class="info-item">Họ tên:</div>
                <div class="info-item" id="student-name">Đang nhận diện...</div>
                <div class="info-item">Thời gian điểm danh:</div>
                <div class="info-item" id="attendance-time">--:--:--</div>
            </div>
            <script>
                const studentName = document.getElementById('student-name');
                const attendanceTime = document.getElementById('attendance-time');

                // Function to fetch recognition details from the server
                function fetchRecognitionData() {
                    fetch('/recognition_data')
                    .then(response => response.json())
                    .then(data => {
                        if (data.name) {
                        studentName.textContent = data.name;
                        attendanceTime.textContent = data.time;
                        } else {
                        studentName.textContent = "Unknown";
                        attendanceTime.textContent = "--:--:--";
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching recognition data:', error);
                    });
                }

                // Fetch recognition data every 5 seconds
                setInterval(fetchRecognitionData, 5000);
            </script>
        </body>
    </html>
    '''
    # return render_template('recog.html')

if __name__ == "__main__":
    app.run(debug=True)
