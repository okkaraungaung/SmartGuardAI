import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
from config import Config
from database.db import (
    init_db,
    get_settings,
    update_settings,
    add_user,
    get_all_users,
    delete_user,
    get_security_events,
    get_dashboard_stats
)
from modules.object_detection import ObjectDetector
from modules.face_recognition import FaceRecognizer
from modules.camera_stream import LiveCameraManager
from modules.telegram_bot import send_telegram_alert
import base64
import numpy as np
import json

app = Flask(__name__)
app.config.from_object(Config)

for folder in [
    app.config["UPLOAD_FOLDER"],
    app.config["FACES_FOLDER"],
    app.config["PROCESSED_FOLDER"],
    app.config["SNAPSHOT_FOLDER"],
    os.path.dirname(app.config["DATABASE_PATH"])
]:
    os.makedirs(folder, exist_ok=True)

DEFAULT_SETTINGS = {
    "ip_camera_url": "http://192.168.1.100:8080/video",
    "telegram_bot_token": app.config["TELEGRAM_BOT_TOKEN"],
    "telegram_chat_id": app.config["TELEGRAM_CHAT_ID"],
    "loitering_threshold": str(app.config["DEFAULT_LOITERING_SECONDS"]),
    "detection_confidence": str(app.config["DETECTION_CONFIDENCE"]),
    "enable_unknown": "1",
    "enable_loitering": "1",
    "enable_package": "1",
    "enable_vehicle": "1"
}

init_db(app.config["DATABASE_PATH"], DEFAULT_SETTINGS)

detector = ObjectDetector(
    model_path=app.config["YOLO_MODEL_PATH"],
    target_classes=app.config["TARGET_CLASSES"],
    conf_threshold=app.config["DETECTION_CONFIDENCE"]
)

face_recognizer = FaceRecognizer(
    yunet_path=app.config["YUNET_MODEL_PATH"],
    sface_path=app.config["SFACE_MODEL_PATH"],
    similarity_threshold=app.config["FACE_SIMILARITY_THRESHOLD"]
)

camera_manager = LiveCameraManager(
    detector=detector,
    face_recognizer=face_recognizer,
    snapshot_dir=app.config["SNAPSHOT_FOLDER"],
    db_path=app.config["DATABASE_PATH"],
    config=app.config
)


def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# Page 1: Dashboard Overview
@app.route("/")
def dashboard():
    stats = get_dashboard_stats(app.config["DATABASE_PATH"])
    return render_template("dashboard.html", stats=stats)


# Page 2: Live Smartphone IP Camera Monitor
@app.route("/monitor")
def monitor():
    settings = get_settings(app.config["DATABASE_PATH"])
    return render_template(
        "monitor.html",
        settings=settings,
        is_running=camera_manager.running,
        status_message=camera_manager.status_message,
        force_enhance=camera_manager.force_enhance
    )


@app.route("/monitor/start", methods=["POST"])
def start_monitor():
    source_type = request.form.get("source_type", "ip_cam")
    ip_url = request.form.get("ip_camera_url", "").strip()
    force_enhance = request.form.get("force_enhance") == "on"

    if source_type == "webcam":
        source = "0"
        label = "MacBook Webcam"
    else:
        source = ip_url
        label = f"Smartphone IP Cam ({ip_url})"
        update_settings(app.config["DATABASE_PATH"], {"ip_camera_url": ip_url})

    registered_users = get_all_users(app.config["DATABASE_PATH"])
    runtime_settings = get_settings(app.config["DATABASE_PATH"])

    ok = camera_manager.start(
        source=source,
        source_label=label,
        registered_users=registered_users,
        runtime_settings=runtime_settings,
        force_enhance=force_enhance
    )
    if ok:
        flash(f"Connected to {label}. Live monitoring started!", "success")
    else:
        flash(f"Could not open stream at '{source}'. Check that IP Webcam is running and on the same Wi-Fi.", "danger")

    return redirect(url_for("monitor"))


@app.route("/monitor/stop", methods=["POST"])
def stop_monitor():
    camera_manager.stop()
    flash("Live camera stream stopped.", "info")
    return redirect(url_for("monitor"))


@app.route("/monitor/night", methods=["POST"])
def toggle_night():
    camera_manager.set_night_mode(not camera_manager.force_enhance)
    return jsonify({"force_enhance": camera_manager.force_enhance})


@app.route("/monitor/status")
def monitor_status():
    return jsonify({
        "running": camera_manager.running,
        "status": camera_manager.status_message,
        "tracks": camera_manager.active_tracks,
        "events": camera_manager.recent_live_events
    })

@app.route("/monitor/roi", methods=["POST"])
def update_monitor_roi():
    data = request.get_json(silent=True) or {}
    roi_enabled = bool(data.get("roi_enabled", True))
    polygon = data.get("polygon")

    updates = {"roi_enabled": "1" if roi_enabled else "0"}
    if polygon and isinstance(polygon, list) and len(polygon) >= 3:
        updates["roi_polygon"] = json.dumps(polygon)

    update_settings(app.config["DATABASE_PATH"], updates)
    camera_manager.update_roi(roi_enabled, polygon)

    return jsonify({"ok": True})

@app.route("/video_feed")
def video_feed():
    if not camera_manager.running:
        return Response(status=204)
    return Response(
        camera_manager.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# Page 3: Visitor Management
@app.route("/visitors", methods=["GET", "POST"])
def visitors():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        photo = request.files.get("photo")

        if not name or not photo or photo.filename == "":
            flash("Please provide both a name and a clear front-facing photo.", "warning")
            return redirect(url_for("visitors"))

        if not allowed_file(photo.filename, app.config["ALLOWED_IMAGE_EXTENSIONS"]):
            flash("Invalid image format. Use JPG or PNG.", "danger")
            return redirect(url_for("visitors"))

        filename = secure_filename(f"{name.lower().replace(' ', '_')}_{photo.filename}")
        save_path = os.path.join(app.config["FACES_FOLDER"], filename)
        photo.save(save_path)

        img = cv2.imread(save_path)
        embedding = face_recognizer.extract_embedding(img)
        if embedding is None:
            os.remove(save_path)
            flash("No face detected in the uploaded image. Upload a clearer portrait.", "danger")
            return redirect(url_for("visitors"))

        add_user(app.config["DATABASE_PATH"], name, filename, embedding)
        flash(f"Registered family member: {name}", "success")
        return redirect(url_for("visitors"))

    users = get_all_users(app.config["DATABASE_PATH"])
    return render_template("visitors.html", users=users)

@app.route("/visitors/check_pose", methods=["POST"])
def check_pose():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame", "")
    target_pose = data.get("target_pose", "radial")
    base_yaw = float(data.get("base_yaw", 0.0))
    base_pitch = float(data.get("base_pitch", 0.55))

    if target_pose == "ip_cam":
        with camera_manager.frame_lock:
            raw = None if camera_manager.latest_raw_frame is None else camera_manager.latest_raw_frame.copy()
        if raw is None:
            return jsonify({"detected": False, "matched": False, "message": "Live IP camera is not running."})
        res = face_recognizer.analyze_face_pose(raw)
        if not res["detected"] or res["too_far"]:
            return jsonify({"detected": res["detected"], "too_far": res["too_far"], "matched": False})
        _, buf = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        preview_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
        return jsonify({
            "detected": True,
            "too_far": False,
            "matched": True,
            "embedding": res["embedding"],
            "preview_frame": preview_url
        })

    if not frame_b64:
        return jsonify({"detected": False, "matched": False})

    try:
        encoded = frame_b64.split(",", 1)[1] if "," in frame_b64 else frame_b64
        img_bytes = base64.b64decode(encoded)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"detected": False, "matched": False})

    res = face_recognizer.analyze_face_pose(img)
    if not res["detected"] or res["too_far"]:
        return jsonify({
            "detected": res["detected"],
            "too_far": res["too_far"],
            "matched": False
        })

    yaw = res["yaw"]
    pitch = res["pitch"]
    dyaw = yaw - base_yaw
    dpitch = pitch - base_pitch

    # Normalize horizontal & vertical head movement onto a 2D unit disk
    nx = dyaw / 0.11       # Positive = user turned left, Negative = user turned right
    ny = dpitch / 0.055    # Negative = chin up, Positive = chin down
    radius = float(np.hypot(nx, ny))

    # Map head direction to 0..360 degrees (0=Right, 90=Up, 180=Left, 270=Down on mirrored screen)
    angle_deg = float((np.degrees(np.arctan2(-ny, -nx)) + 360.0) % 360.0)

    if radius < 0.55:
        sector = "center"
    else:
        # 8 directional sectors of 45 degrees each around the circle
        sector_idx = int(((angle_deg + 22.5) % 360.0) // 45.0)
        sector = f"dir_{sector_idx}"

    return jsonify({
        "detected": True,
        "too_far": False,
        "matched": True,
        "yaw": yaw,
        "pitch": pitch,
        "radius": round(radius, 3),
        "angle_deg": round(angle_deg, 1),
        "sector": sector,
        "embedding": res["embedding"]
    })

@app.route("/visitors/scan", methods=["POST"])
def scan_visitor():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    embeddings = data.get("embeddings", [])
    preview_b64 = data.get("preview_frame", "")

    if not name or len(embeddings) < 4:
        return jsonify({"ok": False, "message": "Please complete the circle scan before saving."}), 400

    saved_preview = secure_filename(f"{name.lower().replace(' ', '_')}_scan.jpg")
    if preview_b64:
        encoded = preview_b64.split(",", 1)[1] if "," in preview_b64 else preview_b64
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(os.path.join(app.config["FACES_FOLDER"], saved_preview), img)

    add_user(app.config["DATABASE_PATH"], name, saved_preview, embeddings)
    return jsonify({
        "ok": True,
        "message": f"Registered {name} with {len(embeddings)} multi-angle Face ID samples!"
    })

@app.route("/visitors/delete/<int:user_id>", methods=["POST"])
def remove_visitor(user_id):
    delete_user(app.config["DATABASE_PATH"], user_id)
    flash("Registered profile removed.", "info")
    return redirect(url_for("visitors"))


# Page 4: Security Event History
@app.route("/events")
def events():
    event_type = request.args.get("event_type", "ALL")
    date_query = request.args.get("date", "").strip()
    event_list = get_security_events(
        app.config["DATABASE_PATH"],
        event_type=event_type,
        date_query=date_query if date_query else None
    )
    return render_template(
        "events.html",
        events=event_list,
        selected_type=event_type,
        selected_date=date_query
    )


# Page 5: Settings
@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        action = request.form.get("action", "save")
        new_settings = {
            "ip_camera_url": request.form.get("ip_camera_url", "").strip(),
            "telegram_bot_token": request.form.get("telegram_bot_token", "").strip(),
            "telegram_chat_id": request.form.get("telegram_chat_id", "").strip(),
            "loitering_threshold": request.form.get("loitering_threshold", "15.0").strip(),
            "detection_confidence": request.form.get("detection_confidence", "0.40").strip(),
            "enable_unknown": "1" if request.form.get("enable_unknown") == "on" else "0",
            "enable_loitering": "1" if request.form.get("enable_loitering") == "on" else "0",
            "enable_package": "1" if request.form.get("enable_package") == "on" else "0",
            "enable_vehicle": "1" if request.form.get("enable_vehicle") == "on" else "0"
        }
        update_settings(app.config["DATABASE_PATH"], new_settings)

        if action == "test_telegram":
            ok = send_telegram_alert(
                bot_token=new_settings["telegram_bot_token"],
                chat_id=new_settings["telegram_chat_id"],
                message="<b>SmartGuard AI Test Notification</b>\nLive IP camera alerting is active."
            )
            if ok:
                flash("Settings saved and test Telegram alert delivered!", "success")
            else:
                flash("Settings saved, but Telegram test failed. Check your Bot Token and Chat ID.", "danger")
        else:
            flash("System settings updated.", "success")

        return redirect(url_for("settings"))

    current_settings = get_settings(app.config["DATABASE_PATH"])
    return render_template("settings.html", settings=current_settings)


@app.route("/uploads/faces/<filename>")
def serve_face(filename):
    return send_from_directory(app.config["FACES_FOLDER"], filename)


@app.route("/results/snapshots/<filename>")
def serve_snapshot(filename):
    return send_from_directory(app.config["SNAPSHOT_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)