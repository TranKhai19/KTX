from flask import Flask, render_template, Response, jsonify
import pickle
import face_recognition
import cv2
import imutils
import time

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

recognized_name = "Unknown"
attendance_time = "--:--:--"

# Load dữ liệu mã hóa khuôn mặt từ file pickle
with open('encodings.pickle', "rb") as file:
    data = pickle.load(file)

# Khởi tạo camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Sử dụng DirectShow cho Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

# Kiểm tra camera có hoạt động không
if not cap.isOpened():
    raise RuntimeError("Không thể mở camera, kiểm tra lại kết nối camera!")

# Hàm xử lý phát video từ camera và nhận diện khuôn mặt
def generate_frames():
    global recognized_name, attendance_time  # Sử dụng biến toàn cục
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Không thể đọc frame từ camera.")
            break  # Thoát nếu không thể đọc frame

        # Chuyển đổi frame từ BGR sang RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(rgb, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # Phát hiện khuôn mặt và mã hóa
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # Tính tần suất xuất hiện của từng khuôn mặt
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)

        # Cập nhật thông tin recognized_name và attendance_time
        if names:
            recognized_name = names[0]
            attendance_time = time.strftime("%H:%M:%S")

        # Vẽ khung và tên lên frame
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            bottom = int(bottom * r)
            left = int(left * r)
            right = int(right * r)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Chuyển frame thành định dạng JPEG để trả về
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Đường dẫn cung cấp video feed từ camera
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Cung cấp thông tin nhận diện dưới dạng JSON
@app.route('/recognition_data')
def recognition_data():
    global recognized_name, attendance_time
    return jsonify({
        'name': recognized_name,
        'time': attendance_time
    })

# Trang nhận diện sinh viên
@app.route('/recognition')
def recognition_page():
    return render_template('recognition.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/screen')
def screen():
    return render_template('screen.html')
# Giải phóng camera khi kết thúc ứng dụng
@app.teardown_appcontext
def cleanup(error=None):
    if cap.isOpened():
        cap.release()

if __name__ == '__main__':
    app.run(debug=True)
