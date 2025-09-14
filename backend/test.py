import os
import cv2
import time
import json
import csv
import math
import pickle
import argparse
from datetime import datetime, timedelta
from collections import deque, defaultdict

# Optional dependencies
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except Exception:
    FACE_REC_AVAILABLE = False

import mediapipe as mp
import numpy as np

# --------------------------
# Configuration (tweak here)
# --------------------------
OUTPUT_DIR_RECORD = "recordings"
OUTPUT_DIR_SNAP   = "snapshots"
OUTPUT_DIR_LOGS   = "logs"
ENROLLED_DIR      = "enrolled"   # stores encodings per candidate
CAM_INDEX         = 0            # default webcam
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
FPS_TARGET        = 30

# Head pose thresholds (degrees)
YAW_LIMIT   = 25.0   # left-right
PITCH_LIMIT = 20.0   # up-down

# Mouth open threshold (ratio of mouth height to face size)
MOUTH_OPEN_THRESH = 0.065

# Hands near face threshold (in pixels, distance between hand landmarks and face bbox)
HAND_NEAR_FACE_PIXELS = 120

# Event cooldowns (seconds) to reduce spam
COOLDOWN_SAME_EVENT = 10.0

# Draw / Recording
DRAW_LANDMARKS = True
RECORD_ANNOTATED_VIDEO = True
VIDEO_CODEC = "MJPG"  # widely compatible on Windows (creates .avi)

# --------------------------
# Utility helpers
# --------------------------
def ensure_dirs():
    os.makedirs(OUTPUT_DIR_RECORD, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SNAP, exist_ok=True)
    os.makedirs(OUTPUT_DIR_LOGS, exist_ok=True)
    os.makedirs(ENROLLED_DIR, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def path_safe(s):
    return "".join(c for c in s if c.isalnum() or c in ("-", "_"))

def save_snapshot(frame, label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{path_safe(label)}_{ts}.jpg"
    p = os.path.join(OUTPUT_DIR_SNAP, fname)
    try:
        cv2.imwrite(p, frame)
        return p
    except Exception as e:
        print("Snapshot save error:", e)
        return None

def write_event(csv_writer, event_type, details, t=None):
    if t is None:
        t = timestamp()
    csv_writer.writerow({"time": t, "event": event_type, "details": details})

# --------------------------
# Identity enroll/verify
# --------------------------
def enroll_candidate(name, cap, frames_to_collect=12, step=2):
    """
    Capture several frames and build an encoding set for the candidate.
    """
    if not FACE_REC_AVAILABLE:
        print("face_recognition not available; cannot enroll.")
        return False

    print(f"Starting enrollment for: {name}")
    print("Look at the camera. Press 'q' to cancel.")

    encs = []
    collected = 0
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        view = frame.copy()
        cv2.putText(view, f"Enrolling {name} ({collected}/{frames_to_collect})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("Enroll", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Enrollment canceled by user.")
            break

        if frame_id % max(1, step) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if locs:
                enc = face_recognition.face_encodings(rgb, locs)[0]
                encs.append(enc)
                collected += 1
                if collected >= frames_to_collect:
                    break
        frame_id += 1

    cv2.destroyWindow("Enroll")

    if collected == 0:
        print("No face captured; enrollment failed.")
        return False

    enc_path = os.path.join(ENROLLED_DIR, f"{path_safe(name)}_encodings.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(encs, f)
    print(f"Enrollment complete. Saved encodings: {enc_path}")
    return True

def load_candidate_encodings(name):
    enc_path = os.path.join(ENROLLED_DIR, f"{path_safe(name)}_encodings.pkl")
    if not os.path.exists(enc_path):
        return None
    try:
        with open(enc_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def verify_identity(frame, enrolled_encs, tolerance=0.5):
    """
    Return True if the current face matches any of the enrolled encodings.
    """
    if not FACE_REC_AVAILABLE:
        return True  # skip if lib missing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return False
    encs = face_recognition.face_encodings(rgb, locs)
    for e in encs:
        dists = face_recognition.face_distance(enrolled_encs, e)
        if len(dists) and dists.min() <= tolerance:
            return True
    return False

# --------------------------
# Head pose via Face Mesh
# --------------------------
# 3D model points for a rough PnP head pose (nose tip, chin, eye corners, mouth corners)
# Using standard approximate coords in mm.
MODEL_POINTS_3D = np.array([
    [0.0,   0.0,   0.0],    # Nose tip
    [0.0, -63.6, -12.5],    # Chin
    [-43.3, 32.7, -26.0],   # Left eye left corner
    [43.3, 32.7, -26.0],    # Right eye right corner
    [-28.9,-28.9,-24.1],    # Left Mouth corner
    [28.9,-28.9,-24.1],     # Right mouth corner
], dtype=np.float64)

# Face Mesh landmark indices roughly corresponding to the above points:
# (nose tip, chin, left eye corner, right eye corner, left mouth, right mouth)
LMK = dict(nose=1, chin=152, eye_l=33, eye_r=263, mouth_l=61, mouth_r=291)

def solve_head_pose(landmarks, img_w, img_h):
    """
    landmarks: list of (x,y) normalized coords [0,1]
    returns yaw, pitch (degrees), and 2D image points used
    """
    pts = []
    for k in ["nose","chin","eye_l","eye_r","mouth_l","mouth_r"]:
        lm = landmarks[LMK[k]]
        pts.append([lm.x * img_w, lm.y * img_h])
    image_points_2d = np.array(pts, dtype=np.float64)

    # Camera intrinsics approximation
    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points_2d,
                                  camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, image_points_2d

    # Convert rotation vector to Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(-rmat[2,0], sy))
        yaw   = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
        # roll  = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
    else:
        pitch = math.degrees(math.atan2(-rmat[2,0], sy))
        yaw   = 0.0
    return yaw, pitch, image_points_2d

def mouth_open_ratio(landmarks, img_w, img_h):
    # Upper lip 13, lower lip 14 (approx) in FaceMesh
    upper = landmarks[13]
    lower = landmarks[14]
    dy = abs((lower.y - upper.y) * img_h)
    # Normalize by face height (chin 152 to forehead-ish 10)
    chin = landmarks[152]; brow = landmarks[10]
    face_h = abs((chin.y - brow.y) * img_h) + 1e-6
    return dy / face_h

def bbox_from_landmarks(landmarks, img_w, img_h, pad=20):
    xs = [int(l.x * img_w) for l in landmarks]
    ys = [int(l.y * img_h) for l in landmarks]
    x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
    x2, y2 = min(img_w-1, max(xs)+pad), min(img_h-1, max(ys)+pad)
    return x1, y1, x2, y2

# --------------------------
# Main proctoring loop
# --------------------------
def run_proctor(candidate_name=None, duration_min=None):
    ensure_dirs()

    # Identity encodings
    enrolled_encs = None
    if candidate_name and FACE_REC_AVAILABLE:
        enrolled_encs = load_candidate_encodings(candidate_name)
        if enrolled_encs is None:
            print("No encodings found for the candidate; continuing without ID check.")
    elif candidate_name and not FACE_REC_AVAILABLE:
        print("face_recognition not installed. Skipping ID check.")

    # Video IO
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Output recording
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUT_DIR_RECORD, f"session_{session_id}.avi")
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = None
    if RECORD_ANNOTATED_VIDEO:
        writer = cv2.VideoWriter(video_path, fourcc, float(FPS_TARGET), (FRAME_WIDTH, FRAME_HEIGHT))

    # Logs
    log_csv_path = os.path.join(OUTPUT_DIR_LOGS, f"events_{session_id}.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["time", "event", "details"])
    csv_writer.writeheader()

    session_start = datetime.now()
    session_end = None

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # State / cooldowns
    paused = False
    last_event_time = defaultdict(lambda: 0.0)
    presence_missing_since = None
    verified_once = False

    # Duration
    end_time = session_start + timedelta(minutes=duration_min) if duration_min else None

    print("Proctoring started. Press 'q' to quit, 'p' to pause/resume, 's' to snapshot.")

    while True:
        if end_time and datetime.now() >= end_time:
            print("Duration reached; stopping.")
            break

        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        if paused:
            info = "PAUSED - press 'p' to resume"
            cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.imshow("AI Proctor", frame)
            if writer: writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("p"): paused = False
            if key == ord("s"): save_snapshot(frame, "manual_snapshot")
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Mesh inference
        fm_res = face_mesh.process(rgb)
        face_count = 0
        yaw_pitch_list = []
        face_boxes = []

        if fm_res.multi_face_landmarks:
            face_count = len(fm_res.multi_face_landmarks)

            for f_lm in fm_res.multi_face_landmarks:
                yaw, pitch, pts2d = solve_head_pose(f_lm.landmark, w, h)
                if yaw is not None:
                    yaw_pitch_list.append((yaw, pitch))
                # Build face bbox
                x1,y1,x2,y2 = bbox_from_landmarks(f_lm.landmark, w, h)
                face_boxes.append((x1,y1,x2,y2))

                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        frame,
                        f_lm,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(80,80,80))
                    )

        # Hands inference
        hands_res = hands.process(rgb)
        hand_points = []
        if hands_res.multi_hand_landmarks:
            for hlm in hands_res.multi_hand_landmarks:
                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 100, 255)),
                        mp_drawing.DrawingSpec(thickness=2, color=(0, 200, 255))
                    )
                for p in hlm.landmark:
                    hand_points.append((int(p.x * w), int(p.y * h)))

        # Presence check
        if face_count == 0:
            if presence_missing_since is None:
                presence_missing_since = time.time()
            elif time.time() - presence_missing_since >= 2.0:  # away for >=2s
                if time.time() - last_event_time["no_face"] > COOLDOWN_SAME_EVENT:
                    write_event(csv_writer, "no_face", "Candidate not visible")
                    snap = save_snapshot(frame, "no_face")
                    if snap:
                        write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                    last_event_time["no_face"] = time.time()
            status_text = "No face detected"
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 255), 2)
        else:
            presence_missing_since = None

        # Multiple faces
        if face_count > 1:
            if time.time() - last_event_time["multi_face"] > COOLDOWN_SAME_EVENT:
                write_event(csv_writer, "multiple_faces", f"Detected {face_count} faces")
                snap = save_snapshot(frame, "multiple_faces")
                if snap:
                    write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
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
                    if time.time() - last_event_time["off_screen"] > COOLDOWN_SAME_EVENT and face_count > 0:
                        write_event(csv_writer, "off_screen", f"yaw={yaw:.1f}, pitch={pitch:.1f}")
                        snap = save_snapshot(frame, "off_screen")
                        if snap:
                            write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                        last_event_time["off_screen"] = time.time()

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(30, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mouth open / talking heuristic (check primary face 0)
        if fm_res.multi_face_landmarks:
            lms = fm_res.multi_face_landmarks[0].landmark
            m_ratio = mouth_open_ratio(lms, w, h)
            if m_ratio > MOUTH_OPEN_THRESH:
                if time.time() - last_event_time["talking"] > COOLDOWN_SAME_EVENT:
                    write_event(csv_writer, "talking", f"mouth_ratio={m_ratio:.3f}")
                    snap = save_snapshot(frame, "talking")
                    if snap:
                        write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                    last_event_time["talking"] = time.time()
                cv2.putText(frame, "Talking suspected", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

        # Hands near face (possible phone)
        if face_boxes and hand_points:
            # check any hand point within threshold of any face bbox
            near = False
            for (x1,y1,x2,y2) in face_boxes:
                for (hx,hy) in hand_points:
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2
                    d = math.hypot(hx - cx, hy - cy)
                    if d <= HAND_NEAR_FACE_PIXELS:
                        near = True
                        break
                if near: break

            if near:
                if time.time() - last_event_time["hands_near_face"] > COOLDOWN_SAME_EVENT:
                    write_event(csv_writer, "hands_near_face", f"distance<= {HAND_NEAR_FACE_PIXELS}px")
                    snap = save_snapshot(frame, "hands_near_face")
                    if snap:
                        write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                    last_event_time["hands_near_face"] = time.time()
                cv2.putText(frame, "Hands near face", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 160, 0), 2)

        # Identity verification (one-time or periodic)
        if candidate_name and FACE_REC_AVAILABLE and enrolled_encs is not None and face_count > 0:
            if not verified_once:
                ok = verify_identity(frame, enrolled_encs, tolerance=0.5)
                if ok:
                    verified_once = True
                    write_event(csv_writer, "identity_verified", f"candidate={candidate_name}")
                else:
                    if time.time() - last_event_time["identity_mismatch"] > COOLDOWN_SAME_EVENT:
                        write_event(csv_writer, "identity_mismatch", "face does not match enrollment")
                        snap = save_snapshot(frame, "identity_mismatch")
                        if snap:
                            write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")
                        last_event_time["identity_mismatch"] = time.time()
                    cv2.putText(frame, "Identity mismatch", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2)

        # HUD
        hud = f"Faces: {face_count}"
        if candidate_name:
            hud += f" | Candidate: {candidate_name} {'(verified)' if verified_once else '(verifying...)'}"
        if end_time:
            remain = int((end_time - datetime.now()).total_seconds())
            hud += f" | Time left: {remain//60:02d}:{remain%60:02d}"
        cv2.putText(frame, hud, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Show / record
        cv2.imshow("AI Proctor", frame)
        if writer: writer.write(frame)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = True
        elif key == ord("s"):
            snap = save_snapshot(frame, "manual_snapshot")
            if snap:
                write_event(csv_writer, "snapshot", f"Saved {os.path.basename(snap)}")

    # Cleanup
    session_end = datetime.now()
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    csv_file.close()

    # Build summary JSON
    summary_path = os.path.join(OUTPUT_DIR_LOGS, f"summary_{session_id}.json")
    summary = {
        "session_id": session_id,
        "candidate": candidate_name or "unknown",
        "start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": session_end.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": int((session_end - session_start).total_seconds()),
        "video_path": video_path if RECORD_ANNOTATED_VIDEO else None,
        "events_csv": log_csv_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Session complete.")
    print("Video:", summary["video_path"])
    print("Events CSV:", summary["events_csv"])
    print("Summary JSON:", summary_path)

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-Powered Proctoring System")
    parser.add_argument("--enroll", type=str, help="Enroll a candidate by name (uses webcam)")
    parser.add_argument("--candidate", type=str, help="Candidate name to verify against enrolled encodings")
    parser.add_argument("--duration", type=int, default=None, help="Exam duration in minutes")
    args = parser.parse_args()

    ensure_dirs()

    if args.enroll:
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        ok = enroll_candidate(args.enroll, cap)
        cap.release()
        cv2.destroyAllWindows()
        return

    run_proctor(candidate_name=args.candidate, duration_min=args.duration)

if __name__ == "__main__":
    main()
