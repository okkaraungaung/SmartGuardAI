import os
import time
import traceback
import threading
import cv2
from datetime import datetime
from modules.image_processing import enhance_frame
from modules.person_tracking import PersonTrackerManager
from modules.suspicious_activity import SecurityEventEngine
from modules.telegram_bot import send_telegram_alert
from database.db import insert_video, insert_detections_bulk, insert_security_event


class LiveCameraManager:
    def __init__(self, detector, face_recognizer, snapshot_dir, db_path, config):
        self.detector = detector
        self.face_recognizer = face_recognizer
        self.snapshot_dir = snapshot_dir
        self.db_path = db_path
        self.config = config

        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.det_lock = threading.Lock()

        self.latest_raw_frame = None
        self.latest_detections = []
        self.latest_jpeg = None

        self.source_label = "Stopped"
        self.force_enhance = False
        self.status_message = "Camera offline"
        self.active_tracks = []
        self.recent_live_events = []

        self._reader_thread = None
        self._ai_thread = None
        self._render_thread = None

    def start(self, source, source_label, registered_users, runtime_settings, force_enhance=False):
        self.stop()

        raw_source = str(source).strip()
        candidates = [int(raw_source)] if raw_source.isdigit() else [raw_source]
        if not raw_source.isdigit() and raw_source.endswith("/video"):
            candidates.append(raw_source[:-6] + "/videofeed")

        cap = None
        for candidate in candidates:
            test_cap = cv2.VideoCapture(candidate)
            if test_cap.isOpened():
                cap = test_cap
                if not raw_source.isdigit():
                    source_label = f"Smartphone IP Cam ({candidate})"
                break
            test_cap.release()

        if cap is None:
            self.status_message = f"Failed to connect to camera source: {source}"
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self.source_label = source_label
        self.force_enhance = force_enhance
        self.running = True
        self.status_message = f"Live — {source_label}"
        self.latest_raw_frame = None
        self.latest_detections = []
        self.active_tracks = []
        self.recent_live_events = []

        self._reader_thread = threading.Thread(target=self._grab_frames_loop, daemon=True)
        self._ai_thread = threading.Thread(
            target=self._ai_worker_loop,
            args=(registered_users, runtime_settings),
            daemon=True
        )
        self._render_thread = threading.Thread(target=self._render_stream_loop, daemon=True)

        self._reader_thread.start()
        self._ai_thread.start()
        self._render_thread.start()
        return True

    def stop(self):
        self.running = False
        for t in (self._reader_thread, self._ai_thread, self._render_thread):
            if t and t.is_alive():
                t.join(timeout=1.5)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_message = "Camera stopped"

    def set_night_mode(self, enabled):
        self.force_enhance = bool(enabled)

    def _grab_frames_loop(self):
        """Thread 1: Continuously drains camera buffer so latest_raw_frame has zero lag."""
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            with self.frame_lock:
                self.latest_raw_frame = frame

    def _render_stream_loop(self):
        """Thread 2: Smoothly draws fresh AI boxes onto live frames at ~25 FPS."""
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            _, enh_frame, stats = enhance_frame(
                frame=frame,
                target_width=self.config["TARGET_WIDTH"],
                kernel_size=self.config["GAUSSIAN_KERNEL"],
                gamma=self.config["GAMMA_VALUE"],
                clip_limit=self.config["CLAHE_CLIP_LIMIT"],
                tile_grid_size=self.config["CLAHE_TILE_GRID"],
                low_light_threshold=self.config["LOW_LIGHT_THRESHOLD"],
                force_enhance=self.force_enhance
            )

            with self.det_lock:
                age = time.time() - getattr(self, "last_det_time", 0.0)
                detections_copy = list(self.latest_detections) if age < 0.45 else []

            annotated = self.detector.draw_detections(enh_frame, detections_copy)

            mode_tag = "NIGHT-ENHANCED" if (stats["is_low_light"] or self.force_enhance) else "LIVE"
            cv2.putText(
                annotated,
                f"REC [{mode_tag}] | {datetime.now().strftime('%H:%M:%S')} | Det: {len(detections_copy)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                with self.frame_lock:
                    self.latest_jpeg = buffer.tobytes()

            time.sleep(0.035)

    def _dispatch_telegram_async(self, bot_token, chat_id, alert_text, snapshot_filepath, video_id, ev):
        def _send():
            sent = send_telegram_alert(
                bot_token=bot_token,
                chat_id=chat_id,
                message=alert_text,
                image_path=snapshot_filepath
            )
            try:
                insert_security_event(
                    db_path=self.db_path,
                    video_id=video_id,
                    event_type=ev["event_type"],
                    severity=ev["severity"],
                    description=ev["description"],
                    snapshot_path=ev["snapshot_filename"],
                    video_timestamp=ev["video_timestamp"],
                    telegram_sent=sent
                )
            except Exception as db_err:
                print(f"[SmartGuard DB Error] {db_err}")
            ev["telegram_sent"] = sent
            self.recent_live_events.insert(0, ev)
            self.recent_live_events = self.recent_live_events[:10]

        threading.Thread(target=_send, daemon=True).start()

    def _ai_worker_loop(self, registered_users, runtime_settings):
        """Thread 3: Runs YOLO + ByteTrack + SFace + Event rules asynchronously."""
        try:
            self.detector.conf_threshold = float(
                runtime_settings.get("detection_confidence", self.config["DETECTION_CONFIDENCE"])
            )
            self.detector.reset_tracker()

            tracker_mgr = PersonTrackerManager()
            event_engine = SecurityEventEngine(
                loitering_threshold=float(
                    runtime_settings.get("loitering_threshold", self.config["DEFAULT_LOITERING_SECONDS"])
                ),
                cooldown_seconds=self.config["EVENT_COOLDOWN_SECONDS"],
                enable_unknown=runtime_settings.get("enable_unknown", "1") == "1",
                enable_loitering=runtime_settings.get("enable_loitering", "1") == "1",
                enable_package=runtime_settings.get("enable_package", "1") == "1"
            )

            bot_token = runtime_settings.get("telegram_bot_token", "").strip()
            chat_id = runtime_settings.get("telegram_chat_id", "").strip()

            session_name = f"LiveStream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            video_id = insert_video(self.db_path, session_name, self.source_label, 0.0)

            start_time = time.time()
            last_db_flush = time.time()
            pending_db_records = []

            while self.running:
                try:
                    with self.frame_lock:
                        frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

                    if frame is None:
                        time.sleep(0.05)
                        continue

                    orig_resized, enh_frame, stats = enhance_frame(
                        frame=frame,
                        target_width=self.config["TARGET_WIDTH"],
                        kernel_size=self.config["GAUSSIAN_KERNEL"],
                        gamma=self.config["GAMMA_VALUE"],
                        clip_limit=self.config["CLAHE_CLIP_LIMIT"],
                        tile_grid_size=self.config["CLAHE_TILE_GRID"],
                        low_light_threshold=self.config["LOW_LIGHT_THRESHOLD"],
                        force_enhance=self.force_enhance
                    )

                    elapsed = round(time.time() - start_time, 1)
                    detections = self.detector.detect_and_track(enh_frame)

                    # Use enhanced frame in dark rooms, sharp unblurred frame in normal lighting
                    face_source = enh_frame if (stats["is_low_light"] or self.force_enhance) else orig_resized
                    face_matches = self.face_recognizer.identify_faces_in_frame(
                        face_source, detections, registered_users
                    )

                    for idx, det in enumerate(detections):
                        if det["label"] == "person":
                            tid = det["track_id"]
                            matched_name, score = face_matches.get(idx, ("NoFace", 0.0))
                            resolved_name = tracker_mgr.update(
                                track_id=tid,
                                timestamp=elapsed,
                                bbox=det["bbox"],
                                identity=matched_name,
                                face_score=score
                            )
                            det["person_name"] = resolved_name
                            if tid in tracker_mgr.tracks:
                                info = tracker_mgr.tracks[tid]
                                det["dwell_time"] = round(max(0.0, info["last_seen"] - info["first_seen"]), 1)
                                det["face_score"] = info["best_face_score"]

                        pending_db_records.append({
                            "tracking_id": det["track_id"],
                            "label": det["label"],
                            "person_name": det["person_name"],
                            "confidence": det["confidence"],
                            "timestamp": elapsed
                        })

                    with self.det_lock:
                        self.latest_detections = detections
                        self.last_det_time = time.time()

                    self.active_tracks = tracker_mgr.get_active_summary(elapsed, active_window=1.5)

                    if pending_db_records and (time.time() - last_db_flush) >= 3.0:
                        try:
                            insert_detections_bulk(self.db_path, video_id, pending_db_records)
                            pending_db_records.clear()
                            last_db_flush = time.time()
                        except Exception:
                            pass

                    if detections:
                        annotated_snap = self.detector.draw_detections(enh_frame, detections)
                        new_events = event_engine.evaluate_frame(
                            detections=detections,
                            tracker_mgr=tracker_mgr,
                            annotated_frame=annotated_snap,
                            snapshot_dir=self.snapshot_dir,
                            video_basename=session_name,
                            timestamp=elapsed
                        )

                        for ev in new_events:
                            clock_str = datetime.now().strftime("%H:%M:%S")
                            alert_text = (
                                f"<b>SmartGuard AI Alert — {ev['event_type']}</b>\n"
                                f"• <b>Severity:</b> {ev['severity']}\n"
                                f"• <b>Camera:</b> {self.source_label}\n"
                                f"• <b>Time:</b> {clock_str}\n"
                                f"• <b>Details:</b> {ev['description']}"
                            )
                            self._dispatch_telegram_async(
                                bot_token, chat_id, alert_text, ev["snapshot_filepath"], video_id, ev
                            )

                except Exception:
                    print("[SmartGuard AI Frame Error]:")
                    traceback.print_exc()

                time.sleep(0.08)

        except Exception:
            print("[SmartGuard AI Init Error]:")
            traceback.print_exc()
    
    def generate_mjpeg(self):
        while self.running:
            with self.frame_lock:
                frame_bytes = self.latest_jpeg
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.04)