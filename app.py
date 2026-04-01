import cv2
import os
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# โหลด Model
model = YOLO('yolov8n.pt')

# COCO class IDs สำหรับยานพาหนะเท่านั้น
# 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
VEHICLE_NAMES = {1: 'Bike', 2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}

# ตัวแปรควบคุมระบบ (Global)
current_cap = None
track_data = {}
is_finished = False
frame_skip = 0
processed_frame_count = 0

# ค่า config สำหรับ risk scoring
RISK_MOVE_THRESHOLD = 20       # ลดระยะเคลื่อนที่ขั้นต่ำ (pixel)
RISK_INCREMENT = 2             # เพิ่มคะแนนเร็วขึ้น
RISK_DECAY = 0.2               # ลดช้าลง
RISK_HISTORY_LEN = 8           # ใช้ข้อมูลย้อนหลังน้อยลง
MIN_BOX_AREA = 400             # กรอง bounding box เล็กเกินไป (noise)
MAX_TRACKS = 50                # จำกัดจำนวน track สูงสุด
FPS = 30                       # FPS สำหรับคำนวณความเร็ว


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def upload():
    global current_cap, track_data, is_finished, frame_skip
    file = request.files.get('file')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # รีเซ็ตระบบใหม่ทุกครั้งที่อัปโหลด
        if current_cap:
            current_cap.release()
        current_cap = cv2.VideoCapture(filepath)
        track_data = {}
        is_finished = False
        frame_skip = 0
        processed_frame_count = 0


        return jsonify({"status": "success", "message": "อัปโหลดสำเร็จ!"})
    return jsonify({"status": "error", "message": "ไม่พบไฟล์"})


@app.route('/get_next_frame')
def get_next_frame():
    global current_cap, track_data, is_finished, frame_skip, processed_frame_count

    if current_cap is None or not current_cap.isOpened():
        return jsonify({"finished": True})

    success, frame = current_cap.read()
    if not success:
        is_finished = True
        if current_cap:
            current_cap.release()
        return jsonify({"finished": True})

    frame_skip += 1
    is_processed = frame_skip % 2 == 0

    if not is_processed:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            "finished": False,
            "image": img_base64,
            "stats": get_stats()
        })

    processed_frame_count += 1

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        iou=0.45,
        classes=VEHICLE_CLASSES,
        verbose=False
    )[0]

    active_ids = set()

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, t_id, cls, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = box.astype(int)

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if t_id not in track_data:
                if len(track_data) >= MAX_TRACKS:
                    oldest = min(track_data.items(), key=lambda x: x[1]['last_seen'])
                    del oldest[0]
                track_data[t_id] = {
                    'history_x': [],
                    'history_y': [],
                    'history_time': [],
                    'score': 0,
                    'class': cls,
                    'last_seen': processed_frame_count
                }

            td = track_data[t_id]
            current_time = processed_frame_count * (2 / FPS)
            td['history_x'].append(cx)
            td['history_y'].append(cy)
            td['history_time'].append(current_time)
            td['last_seen'] = processed_frame_count
            active_ids.add(t_id)

            if len(td['history_x']) > 50:
                td['history_x'] = td['history_x'][-50:]
                td['history_y'] = td['history_y'][-50:]
                td['history_time'] = td['history_time'][-50:]

            td['speed'] = 0
            if len(td['history_x']) > 2:
                dx = td['history_x'][-1] - td['history_x'][-2]
                dy = td['history_y'][-1] - td['history_y'][-2]
                dt = td['history_time'][-1] - td['history_time'][-2]
                if dt > 0:
                    distance = (dx**2 + dy**2) ** 0.5
                    td['speed'] = (distance / dt) if dt > 0 else 0

            if len(td['history_x']) > RISK_HISTORY_LEN:
                dx = abs(td['history_x'][-1] - td['history_x'][-RISK_HISTORY_LEN])
                dy = abs(td['history_y'][-1] - td['history_y'][-RISK_HISTORY_LEN])
                move = (dx**2 + dy**2) ** 0.5

                direction_changes = 0
                recent_x = td['history_x'][-RISK_HISTORY_LEN:]
                for i in range(2, len(recent_x)):
                    d1 = recent_x[i-1] - recent_x[i-2]
                    d2 = recent_x[i] - recent_x[i-1]
                    if d1 * d2 < 0 and abs(d1) > 3 and abs(d2) > 3:
                        direction_changes += 1

                if move > RISK_MOVE_THRESHOLD and direction_changes >= 1:
                    td['score'] = min(15, td['score'] + RISK_INCREMENT)
                elif direction_changes >= 2:
                    td['score'] = min(15, td['score'] + RISK_INCREMENT * 0.8)
                else:
                    td['score'] = max(0, td['score'] - RISK_DECAY)
            else:
                td['score'] = max(0, td['score'] - 0.1)

            vehicle_name = VEHICLE_NAMES.get(cls, '?')
            if td['score'] >= 5:
                color = (0, 0, 255)
                label_bg = (0, 0, 180)
            elif td['score'] >= 2:
                color = (0, 180, 255)
                label_bg = (0, 120, 200)
            else:
                color = (0, 220, 100)
                label_bg = (0, 160, 60)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            speed_kmh = td['speed'] * 0.1
            label = f"{vehicle_name} #{t_id} RK:{int(td['score'])} {speed_kmh:.0f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), label_bg, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    stale_ids = [k for k, v in track_data.items()
                 if k not in active_ids and processed_frame_count - v['last_seen'] > 30]
    for sid in stale_ids:
        if sid in track_data:
            del track_data[sid]

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "finished": False,
        "image": img_base64,
        "stats": get_stats(),
        "timestamp": processed_frame_count * (2 / FPS)
    })


def get_stats():
    return {
        "ids": [f"ID {k}" for k in track_data.keys()],
        "scores": [float(v['score']) for v in track_data.values()],
        "speeds": [round(v['speed'] * 0.1, 1) for v in track_data.values()]
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)