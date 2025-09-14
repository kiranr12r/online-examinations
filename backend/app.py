# backend/app.py
from flask import Flask, Response, request, jsonify, stream_with_context
from flask_cors import CORS
import proctoring_system
import json
import threading
import time

app = Flask(__name__)
CORS(app)

# Simple in-memory user store (demo only)
users = {}

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400
    if username in users:
        return jsonify({"success": False, "message": "User already exists"}), 400
    users[username] = password
    return jsonify({"success": True, "message": "Registration successful!"})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if users.get(username) == password:
        return jsonify({"success": True, "message": "Login successful!"})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route("/start-proctor", methods=["POST"])
def start_proctor():
    """
    Optional endpoint to mark session start. Frontend calls this and then navigates to /exam.
    The actual stream is produced when client opens /video_feed?candidate=...
    """
    data = request.get_json() or {}
    candidate = data.get("candidate", "unknown")
    duration = data.get("duration", None)  # minutes, optional
    # We can create an initial log entry here
    ev = {"time": proctoring_system.timestamp_str(), "event": "session_started", "details": f"candidate={candidate}", "candidate": candidate}
    proctoring_system.event_queue.put(ev)
    return jsonify({"message": "Proctoring session initialised", "candidate": candidate})

@app.route("/video_feed")
def video_feed():
    """
    MJPEG stream endpoint. Provide candidate via ?candidate=username (optional).
    The streaming generator will run proctoring logic and push events to event_queue.
    """
    candidate = request.args.get("candidate", "unknown")
    duration = request.args.get("duration", None)
    try:
        duration_min = int(duration) if duration else None
    except Exception:
        duration_min = None

    # stream_with_context ensures the generator can use Flask request context if needed
    return Response(stream_with_context(proctoring_system.generate_stream(candidate_name=candidate, duration_min=duration_min)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/events")
def events():
    """
    Server-Sent Events endpoint. Pops events from the event_queue and streams them to clients.
    Multiple clients can connect; items are pulled in FIFO order.
    """
    def event_stream():
        while True:
            try:
                ev = proctoring_system.event_queue.get(timeout=1.0)
                yield f"data: {json.dumps(ev)}\n\n"
            except Exception:
                # timeout, send a keepalive comment every ~5s to prevent connections closing
                yield ": keep-alive\n\n"
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    print("Starting Flask backend on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
