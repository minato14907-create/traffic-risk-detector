import cv2
import os
import base64
import csv
import io
import math
import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(i) for i in obj]
    return obj


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SNAPSHOT_FOLDER'] = 'snapshots'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SNAPSHOT_FOLDER'], exist_ok=True)

# COCO class IDs สำหรับยานพาหนะ
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
VEHICLE_NAMES = {1: 'Bike', 2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}

# ค่า config
RISK_MOVE_THRESHOLD = 20
RISK_INCREMENT = 2
RISK_DECAY = 0.2
RISK_HISTORY_LEN = 8
MIN_BOX_AREA = 400
MAX_TRACKS = 50
DEFAULT_FPS = 30.0
CONF_THRES = float(os.getenv('CONF_THRES', '0.4'))
IOU_THRES  = float(os.getenv('IOU_THRES',  '0.5'))
IMG_SIZE   = int(os.getenv('IMG_SIZE',   '640'))
MAX_DET    = int(os.getenv('MAX_DET',    '200'))
FRAME_STRIDE = max(1, int(os.getenv('FRAME_STRIDE', '3')))
JPEG_QUALITY = max(30, min(90, int(os.getenv('JPEG_QUALITY', '60'))))
INFER_SCALE  = float(os.getenv('INFER_SCALE', '0.75'))

# ---- Global State ----
current_cap           = None
track_data            = {}
is_finished           = False
is_paused             = False
frame_skip            = 0
processed_frame_count = 0
video_fps             = 30.0
video_width           = 640
video_height          = 480
current_model_name    = 'yolov8s'
model = YOLO(current_model_name)

# ROI polygon (list of [x,y] normalized 0-1)
roi_polygon = []

# Count lines: list of {id, x1,y1,x2,y2 (normalized), count, name}
count_lines = []

# Near-miss log: list of {frame, id1, id2, dist}
near_miss_log = []
NEAR_MISS_DIST = 80   # pixel threshold

# Wrong-way detection: allowed direction angle range (degrees, None = ปิด)
# เช่น allowed_angle_range = (150, 210) = รถควรวิ่งไปทางซ้าย ±30°
wrong_way_config = {'enabled': False, 'min_angle': 0, 'max_angle': 360}
wrong_way_log = []   # {frame, time, track_id}

# Speed alert threshold (pixel/s) — ผู้ใช้ตั้งได้
speed_alert_threshold = 0.0   # 0 = ปิด
pixel_per_meter = 0.0          # 0 = ไม่ calibrate (ใช้ pixel/s * 0.1 แทน)

# Video progress
video_total_frames = 0

# Snapshots: list of {filename, time, reason}
snapshot_log = []

# Heatmap accumulator
heatmap_acc = None

# Event log for CSV export
event_log = []   # list of dicts

# Track which IDs already crossed which line
line_cross_state = {}   # {track_id: {line_id: last_side}}


# ---- Helpers ----

def reset_state():
    global track_data, is_finished, is_paused, frame_skip, processed_frame_count
    global video_fps, video_width, video_height, heatmap_acc
    global near_miss_log, event_log, line_cross_state, snapshot_log, wrong_way_log
    track_data            = {}
    is_finished           = False
    is_paused             = False
    frame_skip            = 0
    processed_frame_count = 0
    near_miss_log         = []
    event_log             = []
    line_cross_state      = {}
    heatmap_acc           = None
    snapshot_log          = []
    wrong_way_log         = []


def point_side(px, py, x1, y1, x2, y2):
    """ด้านของจุด (px,py) เทียบกับเส้น (x1,y1)-(x2,y2) → +1 หรือ -1"""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def check_line_crossing(t_id, cx, cy, frame_w, frame_h):
    """ตรวจว่า track ข้ามเส้นนับไหม"""
    global line_cross_state
    if t_id not in line_cross_state:
        line_cross_state[t_id] = {}
    for line in count_lines:
        lid = line['id']
        x1 = line['x1'] * frame_w
        y1 = line['y1'] * frame_h
        x2 = line['x2'] * frame_w
        y2 = line['y2'] * frame_h
        side = point_side(cx, cy, x1, y1, x2, y2)
        prev = line_cross_state[t_id].get(lid, None)
        if prev is not None and ((prev > 0) != (side > 0)):
            line['count'] += 1
            event_log.append({
                'frame': processed_frame_count,
                'time': round(processed_frame_count * FRAME_STRIDE / video_fps, 2),
                'event': 'line_cross',
                'track_id': t_id,
                'line_name': line.get('name', f'Line{lid}'),
                'detail': ''
            })
        line_cross_state[t_id][lid] = side


def check_near_miss(boxes_centers):
    """ตรวจ near-miss ระหว่างทุกคู่ของยานพาหนะในเฟรม"""
    ids = list(boxes_centers.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            cx1, cy1 = boxes_centers[id1]
            cx2, cy2 = boxes_centers[id2]
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            if dist < NEAR_MISS_DIST:
                near_miss_log.append({
                    'frame': processed_frame_count,
                    'time': round(processed_frame_count * FRAME_STRIDE / video_fps, 2),
                    'id1': int(id1), 'id2': int(id2),
                    'dist': round(dist, 1)
                })
                event_log.append({
                    'frame': processed_frame_count,
                    'time': round(processed_frame_count * FRAME_STRIDE / video_fps, 2),
                    'event': 'near_miss',
                    'track_id': f'{id1}&{id2}',
                    'line_name': '',
                    'detail': f'dist={dist:.1f}px'
                })


def is_in_roi(cx, cy, frame_w, frame_h):
    """ตรวจว่าจุดอยู่ใน ROI polygon ไหม (ถ้าไม่ได้ตั้ง ROI ให้ผ่านทั้งหมด)"""
    if len(roi_polygon) < 3:
        return True
    pts = np.array([[p[0] * frame_w, p[1] * frame_h] for p in roi_polygon], dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)
    return result >= 0


def draw_roi(frame):
    if len(roi_polygon) < 2:
        return
    h, w = frame.shape[:2]
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi_polygon], dtype=np.int32)
    overlay = frame.copy()
    if len(pts) >= 3:
        cv2.fillPoly(overlay, [pts], (59, 130, 246))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], True, (59, 130, 246), 2)
    else:
        cv2.polylines(frame, [pts], False, (59, 130, 246), 2)
    for pt in pts:
        cv2.circle(frame, tuple(pt), 5, (59, 130, 246), -1)


def draw_count_lines(frame):
    h, w = frame.shape[:2]
    for line in count_lines:
        x1 = int(line['x1'] * w)
        y1 = int(line['y1'] * h)
        x2 = int(line['x2'] * w)
        y2 = int(line['y2'] * h)
        cv2.line(frame, (x1, y1), (x2, y2), (245, 158, 11), 2)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        label = f"{line.get('name','Line')} [{line['count']}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (mid_x - 4, mid_y - th - 8), (mid_x + tw + 4, mid_y + 2), (30, 30, 30), -1)
        cv2.putText(frame, label, (mid_x, mid_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 1, cv2.LINE_AA)


def draw_direction_arrow(frame, td, cx, cy, color):
    """วาดลูกศรแสดงทิศทางการเคลื่อนที่"""
    if len(td['history_x']) < 4:
        return
    # ใช้ค่าเฉลี่ยของ 4 เฟรมล่าสุดเพื่อลด noise
    dx = td['history_x'][-1] - td['history_x'][-4]
    dy = td['history_y'][-1] - td['history_y'][-4]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 5:
        return
    # normalize และยืดลูกศร
    length = min(40, max(20, dist * 0.8))
    nx = dx / dist * length
    ny = dy / dist * length
    tip = (int(cx + nx), int(cy + ny))
    cv2.arrowedLine(frame, (int(cx), int(cy)), tip, color, 2, tipLength=0.4)


def get_direction_angle(td):
    """คำนวณมุมทิศทาง (degrees, 0=ขวา, 90=ลง, 180=ซ้าย, 270=ขึ้น)"""
    if len(td['history_x']) < 4:
        return None
    dx = td['history_x'][-1] - td['history_x'][-4]
    dy = td['history_y'][-1] - td['history_y'][-4]
    if math.sqrt(dx*dx + dy*dy) < 5:
        return None
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return angle


def check_wrong_way(t_id, td, frame, cx, cy, current_time):
    """ตรวจสอบรถสวนทาง"""
    if not wrong_way_config['enabled']:
        return False
    angle = get_direction_angle(td)
    if angle is None:
        return False
    mn = wrong_way_config['min_angle']
    mx = wrong_way_config['max_angle']
    # ตรวจว่าอยู่นอก allowed range
    if mn <= mx:
        in_allowed = mn <= angle <= mx
    else:  # wrap around 360
        in_allowed = angle >= mn or angle <= mx
    if not in_allowed:
        if not td.get('wrong_way_alerted'):
            td['wrong_way_alerted'] = True
            wrong_way_log.append({'frame': processed_frame_count, 'time': round(current_time, 2), 'track_id': int(t_id)})
            event_log.append({'frame': processed_frame_count, 'time': round(current_time, 2),
                               'event': 'wrong_way', 'track_id': t_id, 'line_name': '', 'detail': f'angle={angle:.0f}'})
            save_snapshot(frame.copy(), 'wrong_way', t_id)
        return True
    else:
        td['wrong_way_alerted'] = False
        return False


def update_heatmap(cx, cy, frame_w, frame_h):
    global heatmap_acc
    if heatmap_acc is None:
        heatmap_acc = np.zeros((frame_h, frame_w), dtype=np.float32)
    ix = max(0, min(frame_w - 1, int(cx)))
    iy = max(0, min(frame_h - 1, int(cy)))
    cv2.circle(heatmap_acc, (ix, iy), 20, 1.0, -1)


def get_heatmap_base64(frame_w, frame_h):
    if heatmap_acc is None:
        return None
    blurred = cv2.GaussianBlur(heatmap_acc, (51, 51), 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    _, buf = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode('utf-8')


def calc_speed_display(raw_speed_px_per_s):
    """แปลงความเร็ว pixel/s → หน่วยที่แสดง (m/s หรือ km/h ถ้า calibrate, ไม่งั้นใช้ px/s*0.1)"""
    if pixel_per_meter > 0:
        ms = raw_speed_px_per_s / pixel_per_meter
        return round(ms * 3.6, 1)   # km/h จริง
    return round(raw_speed_px_per_s * 0.1, 1)


def save_snapshot(frame, reason, t_id):
    """บันทึกภาพเฟรมที่มีเหตุการณ์สำคัญ"""
    global snapshot_log
    ts = processed_frame_count
    fname = f"snap_{ts}_{reason}_{t_id}.jpg"
    fpath = os.path.join(app.config['SNAPSHOT_FOLDER'], fname)
    cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    snapshot_log.append({
        'filename': fname,
        'frame': ts,
        'time': round(ts * FRAME_STRIDE / video_fps, 2),
        'reason': reason,
        'track_id': t_id
    })
    return fname


def get_stats():
    speeds_display = [calc_speed_display(v.get('speed', 0)) for v in track_data.values()]
    speed_unit = 'km/h' if pixel_per_meter > 0 else 'px/s'
    # speed alert flags
    speed_alerts = []
    if speed_alert_threshold > 0:
        for k, v in track_data.items():
            spd = calc_speed_display(v.get('speed', 0))
            if spd >= speed_alert_threshold:
                speed_alerts.append(int(k))
    return to_native({
        "ids":    [f"ID {k}" for k in track_data.keys()],
        "scores": [float(v['score']) for v in track_data.values()],
        "speeds": speeds_display,
        "speed_unit": speed_unit,
        "speed_alerts": speed_alerts,
        "classes":[VEHICLE_NAMES.get(v['class'], '?') for v in track_data.values()],
        "line_counts": [{'name': line.get('name', f"Line{line['id']}"), 'count': line['count']} for line in count_lines],
        "near_miss_total": len(near_miss_log),
        "recent_near_miss": near_miss_log[-3:] if near_miss_log else [],
        "progress": round(processed_frame_count * FRAME_STRIDE / video_total_frames * 100, 1) if video_total_frames > 0 else 0,
        "snapshot_count": len(snapshot_log),
        "recent_snapshots": snapshot_log[-3:] if snapshot_log else [],
        "wrong_way_total": len(wrong_way_log),
        "wrong_way_config": wrong_way_config
    })


# ---- Routes ----

@app.route('/')
def index():
    return render_template('main.html')


@app.route('/list_uploads')
def list_uploads():
    folder = app.config['UPLOAD_FOLDER']
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    files = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in video_exts:
            path = os.path.join(folder, f)
            files.append({'name': f, 'size': os.path.getsize(path), 'mtime': os.path.getmtime(path)})
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify({'files': files})


@app.route('/load_existing', methods=['POST'])
def load_existing():
    global current_cap, video_fps, video_width, video_height, video_total_frames
    filename = request.json.get('filename', '')
    if not filename:
        return jsonify({'status': 'error', 'message': 'ไม่ระบุชื่อไฟล์'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': 'ไม่พบไฟล์'})
    if current_cap:
        current_cap.release()
    current_cap = cv2.VideoCapture(filepath)
    reset_state()
    fps = current_cap.get(cv2.CAP_PROP_FPS)
    video_fps          = fps if fps and fps > 1 else DEFAULT_FPS
    video_width        = int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height       = int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frames = int(current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return jsonify({'status': 'success', 'message': f'โหลด {filename} สำเร็จ',
                    'fps': video_fps, 'width': video_width, 'height': video_height,
                    'total_frames': video_total_frames})


@app.route('/upload', methods=['POST'])
def upload():
    global current_cap, video_fps, video_width, video_height, video_total_frames
    file = request.files.get('file')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        if current_cap:
            current_cap.release()
        current_cap = cv2.VideoCapture(filepath)
        reset_state()
        fps = current_cap.get(cv2.CAP_PROP_FPS)
        video_fps          = fps if fps and fps > 1 else DEFAULT_FPS
        video_width        = int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height       = int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_total_frames = int(current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return jsonify({"status": "success", "message": "อัปโหลดสำเร็จ!",
                        'fps': video_fps, 'width': video_width, 'height': video_height,
                        'total_frames': video_total_frames})
    return jsonify({"status": "error", "message": "ไม่พบไฟล์"})


@app.route('/set_model', methods=['POST'])
def set_model():
    global model, current_model_name
    name = request.json.get('model', 'yolov8s.pt')
    allowed = {'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'}
    if name not in allowed:
        return jsonify({'status': 'error', 'message': 'model ไม่รองรับ'})
    if name != current_model_name:
        model = YOLO(name)
        current_model_name = name
    return jsonify({'status': 'success', 'model': name})


@app.route('/set_roi', methods=['POST'])
def set_roi():
    global roi_polygon
    roi_polygon = request.json.get('polygon', [])
    return jsonify({'status': 'success', 'points': len(roi_polygon)})


@app.route('/set_lines', methods=['POST'])
def set_lines():
    global count_lines, line_cross_state
    count_lines = request.json.get('lines', [])
    for line in count_lines:
        line.setdefault('count', 0)
    line_cross_state = {}
    return jsonify({'status': 'success', 'lines': len(count_lines)})


@app.route('/pause', methods=['POST'])
def pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({'paused': is_paused})


@app.route('/get_heatmap')
def get_heatmap():
    img = get_heatmap_base64(video_width, video_height)
    return jsonify({'image': img})


@app.route('/export_csv')
def export_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['frame', 'time', 'event', 'track_id', 'line_name', 'detail'])
    writer.writeheader()
    writer.writerows(event_log)
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=traffic_events.csv'})


@app.route('/set_speed_config', methods=['POST'])
def set_speed_config():
    global speed_alert_threshold, pixel_per_meter
    speed_alert_threshold = float(request.json.get('threshold', 0))
    pixel_per_meter       = float(request.json.get('pixel_per_meter', 0))
    return jsonify({'status': 'success', 'threshold': speed_alert_threshold, 'ppm': pixel_per_meter})


@app.route('/set_wrong_way', methods=['POST'])
def set_wrong_way():
    global wrong_way_config
    data = request.json
    wrong_way_config = {
        'enabled':   bool(data.get('enabled', False)),
        'min_angle': float(data.get('min_angle', 0)),
        'max_angle': float(data.get('max_angle', 360))
    }
    return jsonify({'status': 'success', 'config': wrong_way_config})


@app.route('/export_pdf')
def export_pdf():
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Title'],
                                 fontSize=20, textColor=colors.HexColor('#1e293b'),
                                 spaceAfter=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle('sub', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#64748b'),
                                 spaceAfter=4, alignment=TA_CENTER)
    h2_style    = ParagraphStyle('h2', parent=styles['Heading2'],
                                 fontSize=13, textColor=colors.HexColor('#1e40af'),
                                 spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle('body', parent=styles['Normal'],
                                 fontSize=9, textColor=colors.HexColor('#374151'),
                                 spaceAfter=3)

    story = []
    now = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')

    # Title
    story.append(Paragraph('🚨 AI Traffic Risk Analysis Report', title_style))
    story.append(Paragraph(f'สร้างเมื่อ {now} · ระบบ AI Traffic Center v2.0', sub_style))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=12))

    # Summary stats
    scores = [float(v['score']) for v in track_data.values()]
    total  = len(scores)
    safe   = len([s for s in scores if s < 5])
    warn   = len([s for s in scores if 5 <= s < 10])
    danger = len([s for s in scores if s >= 10])
    nm     = len(near_miss_log)
    ww     = len(wrong_way_log)
    snaps  = len(snapshot_log)

    story.append(Paragraph('สรุปผลการวิเคราะห์', h2_style))
    summary_data = [
        ['รายการ', 'จำนวน', 'หมายเหตุ'],
        ['พาหนะที่ตรวจพบทั้งหมด', str(total), 'unique tracks'],
        ['ปลอดภัย (score < 5)', str(safe), ''],
        ['เฝ้าระวัง (score 5–9)', str(warn), 'ควรติดตาม'],
        ['เสี่ยงสูง (score ≥ 10)', str(danger), 'ต้องดำเนินการ'],
        ['Near-Miss events', str(nm), 'ระยะห่างน้อยกว่า 80px'],
        ['Wrong-Way events', str(ww), 'รถสวนทาง'],
        ['Auto Snapshots', str(snaps), 'บันทึกภาพอัตโนมัติ'],
    ]
    t = Table(summary_data, colWidths=[7*cm, 3*cm, 7*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTSIZE',   (0,0), (-1,0), 10),
        ('FONTSIZE',   (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN',      (1,0), (1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Event log table
    if event_log:
        story.append(Paragraph('บันทึกเหตุการณ์ (Event Log)', h2_style))
        ev_data = [['เวลา (s)', 'ประเภทเหตุการณ์', 'Track ID', 'รายละเอียด']]
        event_colors_map = {
            'high_risk': colors.HexColor('#fee2e2'),
            'near_miss': colors.HexColor('#fef3c7'),
            'speed_alert': colors.HexColor('#fef9c3'),
            'wrong_way': colors.HexColor('#fce7f3'),
            'line_cross': colors.HexColor('#f0fdf4'),
        }
        row_bg = []
        for i, ev in enumerate(event_log[:50]):  # max 50 rows
            ev_data.append([
                str(ev.get('time', '')),
                ev.get('event', '').replace('_', ' ').title(),
                str(ev.get('track_id', '')),
                ev.get('detail', '') or ev.get('line_name', '')
            ])
            bg = event_colors_map.get(ev.get('event', ''), colors.white)
            row_bg.append(('BACKGROUND', (0, i+1), (-1, i+1), bg))

        ev_table = Table(ev_data, colWidths=[2.5*cm, 4*cm, 3*cm, 7.5*cm])
        ev_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#374151')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ] + row_bg
        ev_table.setStyle(TableStyle(ev_style))
        story.append(ev_table)
        if len(event_log) > 50:
            story.append(Paragraph(f'* แสดง 50 รายการแรกจากทั้งหมด {len(event_log)} รายการ (ดูครบได้จาก CSV export)', body_style))
        story.append(Spacer(1, 12))

    # Snapshots
    snap_folder = app.config['SNAPSHOT_FOLDER']
    snap_files  = [s for s in snapshot_log if os.path.exists(os.path.join(snap_folder, s['filename']))]
    if snap_files:
        story.append(Paragraph('ภาพเหตุการณ์สำคัญ (Auto Snapshots)', h2_style))
        # แสดงสูงสุด 6 ภาพ
        for i in range(0, min(len(snap_files), 6), 2):
            row_imgs = []
            for j in range(2):
                if i+j < len(snap_files):
                    s = snap_files[i+j]
                    fpath = os.path.join(snap_folder, s['filename'])
                    try:
                        img = RLImage(fpath, width=8*cm, height=5*cm)
                        cap = Paragraph(f"{s['reason'].replace('_',' ').title()} · ID {s['track_id']} · {s['time']}s", body_style)
                        row_imgs.append([img, cap])
                    except Exception:
                        row_imgs.append(['', ''])
                else:
                    row_imgs.append(['', ''])
            tbl = Table([[row_imgs[0][0], row_imgs[1][0]], [row_imgs[0][1], row_imgs[1][1]]],
                        colWidths=[8.5*cm, 8.5*cm])
            tbl.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))

    # Footer
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=12))
    story.append(Paragraph('สร้างโดย AI Traffic Center · YOLOv8 Object Tracking · พัฒนาโดย Thunyathep', sub_style))

    doc.build(story)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='application/pdf',
                    headers={'Content-Disposition': f'attachment; filename=traffic_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.pdf'})


@app.route('/list_snapshots')
def list_snapshots():
    return jsonify({'snapshots': snapshot_log})


@app.route('/get_snapshot/<filename>')
def get_snapshot(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['SNAPSHOT_FOLDER'], filename)


@app.route('/get_next_frame')
def get_next_frame():
    global current_cap, track_data, is_finished, frame_skip, processed_frame_count, video_fps

    if is_paused:
        return jsonify({"finished": False, "paused": True, "stats": get_stats()})

    if current_cap is None or not current_cap.isOpened():
        return jsonify({"finished": True})

    success, frame = current_cap.read()
    if not success:
        is_finished = True
        current_cap.release()
        return jsonify({"finished": True, "stats": get_stats()})

    frame_skip += 1
    is_processed = frame_skip % FRAME_STRIDE == 0

    draw_roi(frame)
    draw_count_lines(frame)

    if not is_processed:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return jsonify({"finished": False, "image": base64.b64encode(buffer).decode('utf-8'), "stats": get_stats()})

    processed_frame_count += 1
    fh, fw = frame.shape[:2]

    infer_frame = frame
    scale_x, scale_y = 1.0, 1.0
    if INFER_SCALE < 1.0:
        infer_frame = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE, interpolation=cv2.INTER_LINEAR)
        scale_x = fw / infer_frame.shape[1]
        scale_y = fh / infer_frame.shape[0]

    try:
        results = model.track(infer_frame, persist=True, conf=CONF_THRES, iou=IOU_THRES,
                               classes=VEHICLE_CLASSES, imgsz=IMG_SIZE, max_det=MAX_DET, verbose=False)[0]
    except Exception:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jsonify({"finished": False, "image": base64.b64encode(buffer).decode('utf-8'), "stats": get_stats()})

    active_ids = set()
    boxes_centers = {}

    if results.boxes.id is not None:
        boxes   = results.boxes.xyxy.cpu().numpy()
        ids     = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, t_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = (int(v) for v in box)
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            x1 = max(0, min(fw - 1, x1))
            x2 = max(0, min(fw - 1, x2))
            y1 = max(0, min(fh - 1, y1))
            y2 = max(0, min(fh - 1, y2))

            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if not is_in_roi(cx, cy, fw, fh):
                continue

            if t_id not in track_data:
                if len(track_data) >= MAX_TRACKS:
                    oldest = min(track_data.items(), key=lambda x: x[1]['last_seen'])
                    del track_data[oldest[0]]
                track_data[t_id] = {
                    'history_x': [], 'history_y': [], 'history_time': [],
                    'score': 0, 'class': cls, 'last_seen': processed_frame_count, 'speed': 0
                }

            td = track_data[t_id]
            current_time = processed_frame_count * (FRAME_STRIDE / video_fps)
            td['history_x'].append(cx)
            td['history_y'].append(cy)
            td['history_time'].append(current_time)
            td['last_seen'] = processed_frame_count
            active_ids.add(t_id)
            boxes_centers[t_id] = (cx, cy)

            if len(td['history_x']) > 50:
                td['history_x'] = td['history_x'][-50:]
                td['history_y'] = td['history_y'][-50:]
                td['history_time'] = td['history_time'][-50:]

            # Speed
            if len(td['history_x']) > 2:
                dx = td['history_x'][-1] - td['history_x'][-2]
                dy = td['history_y'][-1] - td['history_y'][-2]
                dt = td['history_time'][-1] - td['history_time'][-2]
                td['speed'] = ((dx**2 + dy**2) ** 0.5 / dt) if dt > 0 else 0

            # Risk score
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
                    if td['score'] >= 10 and not td.get('snapped_risk'):
                        td['snapped_risk'] = True
                        save_snapshot(frame.copy(), 'high_risk', t_id)
                        event_log.append({'frame': processed_frame_count,
                            'time': round(current_time, 2), 'event': 'high_risk',
                            'track_id': t_id, 'line_name': '', 'detail': f'score={td["score"]:.1f}'})
                    elif td['score'] < 10:
                        td['snapped_risk'] = False
                elif direction_changes >= 2:
                    td['score'] = min(15, td['score'] + RISK_INCREMENT * 0.8)
                else:
                    td['score'] = max(0, td['score'] - RISK_DECAY)
            else:
                td['score'] = max(0, td['score'] - 0.1)

            # Line crossing
            check_line_crossing(t_id, cx, cy, fw, fh)

            # Heatmap
            update_heatmap(cx, cy, fw, fh)

            # Wrong-way detection
            is_wrong_way = check_wrong_way(t_id, td, frame, cx, cy, current_time)

            # Draw box
            if is_wrong_way:
                color, label_bg = (0, 0, 255), (0, 0, 180)
            elif td['score'] >= 10:
                color, label_bg = (0, 0, 255), (0, 0, 180)
            elif td['score'] >= 5:
                color, label_bg = (0, 140, 255), (0, 90, 180)
            elif td['score'] >= 2:
                color, label_bg = (0, 180, 255), (0, 120, 200)
            else:
                color, label_bg = (0, 220, 100), (0, 160, 60)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            speed_display = calc_speed_display(td['speed'])
            speed_unit_str = 'km/h' if pixel_per_meter > 0 else ''
            spd_alert = speed_alert_threshold > 0 and speed_display >= speed_alert_threshold
            if spd_alert:
                color = (0, 0, 255)
                if not td.get('snapped_speed'):
                    td['snapped_speed'] = True
                    save_snapshot(frame.copy(), 'speed_alert', t_id)
                    event_log.append({'frame': processed_frame_count,
                        'time': round(current_time, 2), 'event': 'speed_alert',
                        'track_id': t_id, 'line_name': '', 'detail': f'speed={speed_display}'})
            else:
                td['snapped_speed'] = False
            label = f"{VEHICLE_NAMES.get(cls,'?')} #{t_id} RK:{int(td['score'])} {speed_display}{speed_unit_str}"
            if spd_alert:
                label = '⚡ ' + label
            if is_wrong_way:
                label = '↩ WRONG WAY ' + label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), label_bg, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # Direction arrow
            draw_direction_arrow(frame, td, cx, cy, color)

    # Near-miss check
    if len(boxes_centers) >= 2:
        prev_nm = len(near_miss_log)
        check_near_miss(boxes_centers)
        if len(near_miss_log) > prev_nm:
            nm = near_miss_log[-1]
            save_snapshot(frame.copy(), 'near_miss', f"{nm['id1']}_{nm['id2']}")

    # Draw near-miss warning lines
    nm_recent = [nm for nm in near_miss_log if processed_frame_count - nm['frame'] <= 5]
    for nm in nm_recent:
        if nm['id1'] in boxes_centers and nm['id2'] in boxes_centers:
            p1 = (int(boxes_centers[nm['id1']][0]), int(boxes_centers[nm['id1']][1]))
            p2 = (int(boxes_centers[nm['id2']][0]), int(boxes_centers[nm['id2']][1]))
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(frame, '⚠ NEAR MISS', mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Stale track cleanup
    stale = [k for k, v in track_data.items()
             if k not in active_ids and processed_frame_count - v['last_seen'] > 30]
    for sid in stale:
        del track_data[sid]

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jsonify({
        "finished": False,
        "image": base64.b64encode(buffer).decode('utf-8'),
        "stats": get_stats(),
        "timestamp": processed_frame_count * (FRAME_STRIDE / video_fps),
        "progress": round(processed_frame_count * FRAME_STRIDE / video_total_frames * 100, 1) if video_total_frames > 0 else 0
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
