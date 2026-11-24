import os
import cv2
from modules.image_processing import enhance_frame
from modules.person_tracking import PersonTrackerManager
from modules.suspicious_activity import SecurityEventEngine
from modules.telegram_bot import send_telegram_alert
from database.db import insert_video, insert_detections_bulk, insert_security_event


def create_browser_video_writer(output_path, fps, frame_size):
    for codec in ["avc1", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
    raise RuntimeError("Failed to initialize VideoWriter.")


def process_video_pipeline(
    video_path,
    output_video_dir,
    snapshot_dir,
    detector,
    face_recognizer,
    registered_users,
    runtime_settings,
    config,
    force_enhance=False
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(total_frames / fps, 2) if fps > 0 else 0.0

    target_w = config["TARGET_WIDTH"]
    target_h = int(orig_h * (target_w / float(orig_w)))

    video_filename = os.path.basename(video_path)
    video_basename = os.path.splitext(video_filename)[0]
    processed_filename = f"{video_basename}_tracked.mp4"
    processed_filepath = os.path.join(output_video_dir, processed_filename)

    writer = create_browser_video_writer(processed_filepath, fps, (target_w, target_h))

    # Apply runtime confidence threshold from Settings
    detector.conf_threshold = float(runtime_settings.get("detection_confidence", config["DETECTION_CONFIDENCE"]))
    detector.reset_tracker()

    tracker_mgr = PersonTrackerManager()
    event_engine = SecurityEventEngine(
        loitering_threshold=float(runtime_settings.get("loitering_threshold", config["DEFAULT_LOITERING_SECONDS"])),
        cooldown_seconds=config["EVENT_COOLDOWN_SECONDS"],
        enable_unknown=runtime_settings.get("enable_unknown", "1") == "1",
        enable_loitering=runtime_settings.get("enable_loitering", "1") == "1",
        enable_package=runtime_settings.get("enable_package", "1") == "1",
        enable_vehicle=runtime_settings.get("enable_vehicle", "1") == "1"
    )

    bot_token = runtime_settings.get("telegram_bot_token", "").strip()
    chat_id = runtime_settings.get("telegram_chat_id", "").strip()

    sample_interval = config["FRAME_SAMPLE_INTERVAL"]
    last_detections = []
    db_detection_records = []
    class_counts = {}
    pending_events = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, enh_frame, _ = enhance_frame(
            frame=frame,
            target_width=target_w,
            kernel_size=config["GAUSSIAN_KERNEL"],
            gamma=config["GAMMA_VALUE"],
            clip_limit=config["CLAHE_CLIP_LIMIT"],
            tile_grid_size=config["CLAHE_TILE_GRID"],
            low_light_threshold=config["LOW_LIGHT_THRESHOLD"],
            force_enhance=force_enhance
        )

        if frame_idx % sample_interval == 0:
            timestamp = round(frame_idx / fps, 2)
            detections = detector.detect_and_track(enh_frame)

            for det in detections:
                lbl = det["label"]
                class_counts[lbl] = class_counts.get(lbl, 0) + 1

                if lbl == "person":
                    x1, y1, x2, y2 = det["bbox"]
                    person_crop = enh_frame[max(0, y1):min(target_h, y2), max(0, x1):min(target_w, x2)]
                    matched_name, score = face_recognizer.identify_person_in_crop(person_crop, registered_users)
                    resolved_name = tracker_mgr.update(
                        track_id=det["track_id"],
                        timestamp=timestamp,
                        bbox=det["bbox"],
                        identity=matched_name,
                        face_score=score
                    )
                    det["person_name"] = resolved_name

                db_detection_records.append({
                    "tracking_id": det["track_id"],
                    "label": lbl,
                    "person_name": det["person_name"],
                    "confidence": det["confidence"],
                    "timestamp": timestamp
                })

            last_detections = detections
            annotated_sample = detector.draw_detections(enh_frame, last_detections)

            new_events = event_engine.evaluate_frame(
                detections=detections,
                tracker_mgr=tracker_mgr,
                annotated_frame=annotated_sample,
                snapshot_dir=snapshot_dir,
                video_basename=video_basename,
                timestamp=timestamp
            )
            pending_events.extend(new_events)
        else:
            annotated_sample = detector.draw_detections(enh_frame, last_detections)

        writer.write(annotated_sample)
        frame_idx += 1

    cap.release()
    writer.release()

    video_id = insert_video(config["DATABASE_PATH"], video_filename, video_path, duration)
    insert_detections_bulk(config["DATABASE_PATH"], video_id, db_detection_records)

    # Save security events & dispatch Telegram notifications
    saved_events = []
    for ev in pending_events:
        alert_text = (
            f"<b>SmartGuard AI Alert — {ev['event_type']}</b>\n"
            f"• <b>Severity:</b> {ev['severity']}\n"
            f"• <b>Video:</b> {video_filename}\n"
            f"• <b>Video Time:</b> {ev['video_timestamp']}s\n"
            f"• <b>Details:</b> {ev['description']}"
        )
        sent = send_telegram_alert(
            bot_token=bot_token,
            chat_id=chat_id,
            message=alert_text,
            image_path=ev["snapshot_filepath"]
        )
        insert_security_event(
            db_path=config["DATABASE_PATH"],
            video_id=video_id,
            event_type=ev["event_type"],
            severity=ev["severity"],
            description=ev["description"],
            snapshot_path=ev["snapshot_filename"],
            video_timestamp=ev["video_timestamp"],
            telegram_sent=sent
        )
        ev["telegram_sent"] = sent
        saved_events.append(ev)

    metadata = {
        "video_id": video_id,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "resolution": f"{orig_w}x{orig_h} -> {target_w}x{target_h}",
        "duration": duration
    }

    summary = {
        "processed_video": processed_filename,
        "class_counts": class_counts,
        "visitor_tracks": tracker_mgr.get_summary(),
        "security_events": saved_events
    }

    return metadata, summary