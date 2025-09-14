# backend/proctoring_system.py
import os
import cv2
import time
import json
import csv
import math
import pickle
import argparse
import datetime
from datetime import datetime as dt, timedelta
from collections import defaultdict
import queue

import mediapipe as mp
import numpy as np

# --------------------------
# Configuration (tweak here)
# --------------------------
OUTPUT_DIR_RECORD = "recordings"
OUTPUT_DIR_SNAP   = "snapshots"
OUTPUT_DIR_LOGS   = "logs"
ENROLLED_DIR      = "enrolled"   # (not used heavily in streaming demo)
CAM_INDEX         = 0            # default webcam
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
FPS_TARGET        = 20

# Head pose thresholds (degrees)
YAW_LIMIT   = 25.0   # left-right
PITCH_LIMIT = 20.0   # up-down

# Mouth open threshold (ratio of mouth height to face size)
MOUTH_OPEN_THRESH = 0.065

# Hands near face threshold (in pixels, distance between hand landmarks and face bbox)
HAND_NEAR_FACE_PIXELS = 120

# Event cooldowns (seconds) to reduce spam
COOLDOWN_SAME_EVENT = 8.0

# Draw / Recording
DRAW_LANDMARKS = True
RECORD_ANNOTATED_VIDEO = False
VIDEO_CODEC = "MJPG"  # for .avi if recording

# make sure dirs exist
def ensure_dirs():
    os.makedirs(OUTPUT_DIR_RECORD, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SNAP, exist_ok=True)
    os.makedirs(OUTPUT_DIR_LOGS, exist_ok=True)
    os.makedirs(ENROLLED_DIR, exist_ok=True)

def timestamp_str():
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")

def path_safe(s):
    return "".join(c for c in s if c.isalnum() or c in ("-", "_"))

def save_snapshot(frame, label, candidate="unknown"):
    ts = dt.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{path_safe(candidate)}_{path_safe(label)}_{ts}.jpg"
    p = os.path.join(OUTPUT_DIR_SNAP, fname)
    try:
        cv2.imwrite(p, frame)
        return p
    except Exception as e:
        print("Snapshot save error:", e)
        return None

def write_event_csv(csv_writer, event_type, details, t=None):
    if t is None:
        t = timestamp_str()
    csv_writer.writerow({"time": t, "event": event_type, "details": details})

# Event queue for SSE
event_queue = queue.Queue()

# --------------------------
# Small helper geometry utilities
# --------------------------
MODEL_POINTS_3D = np.array([
    [0.0,   0.0,   0.0],    # Nose tip
    [0.0, -63.6, -12.5],    # Chin
    [-43.3, 32.7, -26.0],   # Left eye left corner
    [43.3, 32.7, -26.0],    # Right eye right corner
    [-28.9,-28.9,-24.1],    # Left Mouth corner
    [28.9,-28.9,-24.1],     # Right mouth corner
], dtype=np.float64)

# Face Mesh landmark indices roughly corresponding to the above points:
LMK = dict(nose=1, chin=152, eye_l=33, eye_r=263, mouth_l=61, mouth_r=291)

def solve_head_pose(landmarks, img_w, img_h):
    try:
        pts = []
        for k in ["nose","chin","eye_l","eye_r","mouth_l","mouth_r"]:
            lm = landmarks[LMK[k]]
            pts.append([lm.x * img_w, lm.y * img_h])
        image_points_2d = np.array(pts, dtype=np.float64)

        focal_length = img_w
        center = (img_w/2, img_h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))

        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, image_points_2d
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        if sy < 1e-6:
            pitch = math.degrees(math.atan2(-rmat[2,0], sy))
            yaw   = 0.0
        else:
            pitch = math.degrees(math.atan2(-rmat[2,0], sy))
            yaw   = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
        return yaw, pitch, image_points_2d
    except Exception:
        return None, None, None

def mouth_open_ratio(landmarks, img_w, img_h):
    try:
        upper = landmarks[13]
        lower = landmarks[14]
        dy = abs((lower.y - upper.y) * img_h)
        chin = landmarks[152]; brow = landmarks[10]
        face_h = abs((chin.y - brow.y) * img_h) + 1e-6
        return dy / face_h
    except Exception:
        return 0.0

def bbox_from_landmarks(landmarks, img_w, img_h, pad=20):
    xs = [int(l.x * img_w) for l in landmarks]
    ys = [int(l.y * img_h) for l in landmarks]
    x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
    x2, y2 = min(img_w-1, max(xs)+pad), min(img_h-1, max(ys)+pad)
    return x1, y1, x2, y2

# --------------------------
# Main streaming proctor loop (yields JPEG frames)
# --------------------------
def generate_stream(candidate_name="unknown", duration_min=None):
    """
    Generator that yields MJPEG frames for Flask. Also detects suspicious events,
    saves snapshots and pushes events into event_queue for SSE.
    """
    ensure_dirs()

    # CSV log writer for this session
    session_id = dt.now().strftime("%Y%m%d_%H%M%S")
    log_csv_path = os.path.join(OUTPUT_DIR_LOGS, f"events_{path_safe(candidate_name)}_{session_id}.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["time", "event", "details"])
    csv_writer.writeheader()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print("Could not open camera.")
        csv_file.close()
        return

    if RECORD_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        video_path = os.path.join(OUTPUT_DIR_RECORD, f"session_{session_id}.avi")
        writer = cv2.VideoWriter(video_path, fourcc, float(FPS_TARGET), (FRAME_WIDTH, FRAME_HEIGHT))
    else:
        writer = None

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # State
    last_event_time = defaultdict(lambda: 0.0)
    presence_missing_since = None

    start_time = dt.now()
    end_time = start_time + timedelta(minutes=duration_min) if duration_min else None

    print(f"Streaming proctor for {candidate_name} started at {start_time}.")
    try:
        while True:
            if end_time and dt.now() >= end_time:
                print("Duration reached; stopping stream generator.")
                break

            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face mesh
            fm_res = face_mesh.process(rgb)
            face_count = 0
            yaw_pitch_list = []
            face_boxes = []

            if fm_res.multi_face_landmarks:
                face_count = len(fm_res.multi_face_landmarks)

                for f_lm in fm_res.multi_face_landmarks:
                    yaw, pitch, _ = solve_head_pose(f_lm.landmark, w, h)
                    if yaw is not None:
                        yaw_pitch_list.append((yaw, pitch))
                    x1,y1,x2,y2 = bbox_from_landmarks(f_lm.landmark, w, h)
                    face_boxes.append((x1,y1,x2,y2))
                    if DRAW_LANDMARKS:
                        mp_drawing.draw_landmarks(frame, f_lm, mp_face_mesh.FACEMESH_TESSELATION)

            # Hands
            hand_points = []
            hands_res = hands.process(rgb)
            if hands_res.multi_hand_landmarks:
                for hlm in hands_res.multi_hand_landmarks:
                    if DRAW_LANDMARKS:
                        mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                    for p in hlm.landmark:
                        hand_points.append((int(p.x * w), int(p.y * h)))

            # Presence check
            if face_count == 0:
                if presence_missing_since is None:
                    presence_missing_since = time.time()
                elif time.time() - presence_missing_since >= 2.0:
                    if time.time() - last_event_time["no_face"] > COOLDOWN_SAME_EVENT:
                        ev = {"time": timestamp_str(), "event": "no_face", "details": "Candidate not visible", "candidate": candidate_name}
                        event_queue.put(ev)
                        write_event_csv(csv_writer, "no_face", "Candidate not visible")
                        snap = save_snapshot(frame, "no_face", candidate_name)
                        if snap:
                            write_event_csv(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                        last_event_time["no_face"] = time.time()
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 255), 2)
            else:
                presence_missing_since = None

            # Multiple faces
            if face_count > 1:
                if time.time() - last_event_time["multi_face"] > COOLDOWN_SAME_EVENT:
                    ev = {"time": timestamp_str(), "event": "multiple_faces", "details": f"Detected {face_count} faces", "candidate": candidate_name}
                    event_queue.put(ev)
                    write_event_csv(csv_writer, "multiple_faces", f"Detected {face_count} faces")
                    snap = save_snapshot(frame, "multiple_faces", candidate_name)
                    if snap:
                        write_event_csv(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                    last_event_time["multi_face"] = time.time()
                cv2.putText(frame, f"Multiple faces: {face_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

            # Head pose / gaze off-screen
            for idx, (x1,y1,x2,y2) in enumerate(face_boxes):
                color = (0, 200, 0)
                label = "OK"
                if idx < len(yaw_pitch_list):
                    yaw, pitch = yaw_pitch_list[idx]
                    if abs(yaw) > YAW_LIMIT or abs(pitch) > PITCH_LIMIT:
                        color = (0, 0, 255)
                        label = f"Off-screen (yaw {yaw:.0f}, pitch {pitch:.0f})"
                        if time.time() - last_event_time["off_screen"] > COOLDOWN_SAME_EVENT:
                            ev = {"time": timestamp_str(), "event": "off_screen", "details": f"yaw={yaw:.1f}, pitch={pitch:.1f}", "candidate": candidate_name}
                            event_queue.put(ev)
                            write_event_csv(csv_writer, "off_screen", f"yaw={yaw:.1f}, pitch={pitch:.1f}")
                            snap = save_snapshot(frame, "off_screen", candidate_name)
                            if snap:
                                write_event_csv(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                            last_event_time["off_screen"] = time.time()
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, max(30, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Mouth open / talking heuristic (primary face)
            if fm_res.multi_face_landmarks:
                lms = fm_res.multi_face_landmarks[0].landmark
                m_ratio = mouth_open_ratio(lms, w, h)
                if m_ratio > MOUTH_OPEN_THRESH:
                    if time.time() - last_event_time["talking"] > COOLDOWN_SAME_EVENT:
                        ev = {"time": timestamp_str(), "event": "talking", "details": f"mouth_ratio={m_ratio:.3f}", "candidate": candidate_name}
                        event_queue.put(ev)
                        write_event_csv(csv_writer, "talking", f"mouth_ratio={m_ratio:.3f}")
                        snap = save_snapshot(frame, "talking", candidate_name)
                        if snap:
                            write_event_csv(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                        last_event_time["talking"] = time.time()
                    cv2.putText(frame, "Talking suspected", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

            # Hands near face (possible phone)
            if face_boxes and hand_points:
                near = False
                for (x1,y1,x2,y2) in face_boxes:
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2
                    for (hx,hy) in hand_points:
                        d = math.hypot(hx - cx, hy - cy)
                        if d <= HAND_NEAR_FACE_PIXELS:
                            near = True
                            break
                    if near: break

                if near:
                    if time.time() - last_event_time["hands_near_face"] > COOLDOWN_SAME_EVENT:
                        ev = {"time": timestamp_str(), "event": "hands_near_face", "details": f"distance<= {HAND_NEAR_FACE_PIXELS}px", "candidate": candidate_name}
                        event_queue.put(ev)
                        write_event_csv(csv_writer, "hands_near_face", f"distance<= {HAND_NEAR_FACE_PIXELS}px")
                        snap = save_snapshot(frame, "hands_near_face", candidate_name)
                        if snap:
                            write_event_csv(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                        last_event_time["hands_near_face"] = time.time()
                    cv2.putText(frame, "Hands near face", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 160, 0), 2)

            # HUD
            hud = f"Faces: {face_count}"
            if end_time:
                remain = int((end_time - dt.now()).total_seconds())
                hud += f" | Time left: {remain//60:02d}:{remain%60:02d}"
            cv2.putText(frame, hud, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Write frame to video if needed
            if writer is not None:
                writer.write(frame)

            # Encode to jpeg and yield
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            writer.release()
        csv_file.close()
        print("Stream ended, resources released.")
