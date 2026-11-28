import os
import json
import cv2
import numpy as np


class SecurityEventEngine:
    def __init__(
        self,
        loitering_threshold=15.0,
        cooldown_seconds=60.0,
        enable_unknown=True,
        enable_loitering=True,
        enable_package=True,
        roi_enabled=True,
        roi_polygon=None
    ):
        self.loitering_threshold = float(loitering_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.enable_unknown = enable_unknown
        self.enable_loitering = enable_loitering
        self.enable_package = enable_package
        self.roi_enabled = bool(roi_enabled)

        # Normalized 4-point doorway trapezoid [[x, y], ...] in [0.0, 1.0]
        self.roi_polygon = roi_polygon or [
            [0.20, 0.35],
            [0.80, 0.35],
            [0.92, 0.98],
            [0.08, 0.98]
        ]

        self.alert_history = {}
        self.generated_events = []

        # Track per-person ROI dwell time: tid -> {"roi_enter_time": float, "in_roi": bool}
        self.roi_dwell_state = {}

        # Track stationary packages for theft/removal detection:
        # slot_id -> {"label": str, "center": (cx, cy), "bbox": [x1,y1,x2,y2], "first_seen": float, "last_seen": float, "armed": bool, "alerted": bool}
        self.watched_packages = {}
        self._pkg_seq = 1
        self.last_unknown_seen_time = -999.0
        self.last_unknown_id = None

    def set_roi(self, enabled, polygon=None):
        self.roi_enabled = bool(enabled)
        if polygon and len(polygon) >= 3:
            self.roi_polygon = polygon

    def get_pixel_polygon(self, frame_shape):
        h, w = frame_shape[:2]
        pts = np.array(
            [[int(pt[0] * w), int(pt[1] * h)] for pt in self.roi_polygon],
            dtype=np.int32
        )
        return pts

    def is_inside_roi(self, bbox, frame_shape):
        if not self.roi_enabled:
            return True
        x1, y1, x2, y2 = bbox
        # Ground-contact foot point (bottom-center of person bounding box)
        foot_x = float((x1 + x2) / 2.0)
        foot_y = float(y2)
        poly = self.get_pixel_polygon(frame_shape)
        return cv2.pointPolygonTest(poly, (foot_x, foot_y), False) >= 0

    def draw_roi_overlay(self, frame):
        if not self.roi_enabled:
            return frame
        poly = self.get_pixel_polygon(frame.shape)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], (255, 200, 0))
        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 200, 0), thickness=2)
        top_pt = tuple(poly[0])
        cv2.putText(
            frame,
            "DOORWAY ROI ZONE",
            (max(8, top_pt[0]), max(20, top_pt[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 220, 0),
            2
        )
        return frame

    def _can_trigger(self, key, timestamp):
        last_time = self.alert_history.get(key)
        if last_time is None or (timestamp - last_time) >= self.cooldown_seconds:
            self.alert_history[key] = timestamp
            return True
        return False

    def _save_snapshot(self, annotated_frame, snapshot_dir, video_basename, event_slug, timestamp):
        filename = f"{video_basename}_{event_slug}_{int(timestamp * 10)}.jpg"
        filepath = os.path.join(snapshot_dir, filename)
        cv2.imwrite(filepath, annotated_frame)
        return filename, filepath

    def evaluate_frame(self, detections, tracker_mgr, annotated_frame, snapshot_dir, video_basename, timestamp):
        new_events = []
        frame_shape = annotated_frame.shape

        # Track whether any Unknown person or Registered family member is currently present
        unknown_present_ids = []
        known_present = False
        for tid, info in tracker_mgr.tracks.items():
            if info["last_seen"] != timestamp:
                continue
            if info["name"] == "Unknown":
                unknown_present_ids.append(tid)
                self.last_unknown_seen_time = timestamp
                self.last_unknown_id = tid
            elif info["name"] != "Scanning...":
                known_present = True

        # 1. Doorway ROI Loitering Check (Unknown Persons Only)
        for tid, info in tracker_mgr.tracks.items():
            if info["last_seen"] != timestamp:
                self.roi_dwell_state.pop(tid, None)
                continue

            in_roi = self.is_inside_roi(info["last_bbox"], frame_shape)
            state = self.roi_dwell_state.get(tid)

            if in_roi:
                if state is None or not state["in_roi"]:
                    self.roi_dwell_state[tid] = {"roi_enter_time": timestamp, "in_roi": True}
                    roi_dwell = 0.0
                else:
                    roi_dwell = round(max(0.0, timestamp - state["roi_enter_time"]), 1)
            else:
                self.roi_dwell_state[tid] = {"roi_enter_time": None, "in_roi": False}
                roi_dwell = 0.0

            info["in_roi"] = in_roi
            info["roi_dwell"] = roi_dwell

            if info["name"] != "Unknown" or not in_roi:
                continue

            if (self.enable_unknown or self.enable_loitering) and roi_dwell > self.loitering_threshold:
                key = f"unknown_loitering_{tid}"
                if self._can_trigger(key, timestamp):
                    snap_name, snap_path = self._save_snapshot(
                        annotated_frame, snapshot_dir, video_basename, f"unknown_loiter_id{tid}", timestamp
                    )
                    new_events.append({
                        "event_type": "Unknown Loitering",
                        "severity": "Critical",
                        "description": f"Unrecognized person (Track ID #{tid}) loitered inside Doorway ROI for {roi_dwell}s (limit: {self.loitering_threshold}s).",
                        "snapshot_filename": snap_name,
                        "snapshot_filepath": snap_path,
                        "video_timestamp": timestamp
                    })

        # 2. Package Arrival & Stationary Anchor Registration
        current_packages = []
        for det in detections:
            lbl = det["label"]
            if lbl in {"backpack", "handbag", "suitcase"}:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                current_packages.append((lbl, (cx, cy), det["bbox"], det["confidence"]))

                if self.enable_package:
                    key = f"package_{lbl}"
                    if self._can_trigger(key, timestamp):
                        snap_name, snap_path = self._save_snapshot(
                            annotated_frame, snapshot_dir, video_basename, f"package_{lbl}", timestamp
                        )
                        new_events.append({
                            "event_type": "Package Detected",
                            "severity": "Info",
                            "description": f"Package/bag ({lbl}, conf {det['confidence']:.2f}) detected at entrance at {timestamp}s.",
                            "snapshot_filename": snap_name,
                            "snapshot_filepath": snap_path,
                            "video_timestamp": timestamp
                        })

        # Match detected packages to spatial anchors (within 75px radius)
        matched_pkg_ids = set()
        for lbl, (cx, cy), bbox, _ in current_packages:
            best_pid = None
            best_dist = 75.0
            for pid, pinfo in self.watched_packages.items():
                if pinfo["label"] != lbl or pinfo["alerted"]:
                    continue
                px, py = pinfo["center"]
                dist = float(np.hypot(cx - px, cy - py))
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid

            if best_pid is not None:
                matched_pkg_ids.add(best_pid)
                pinfo = self.watched_packages[best_pid]
                pinfo["last_seen"] = timestamp
                pinfo["center"] = (cx, cy)
                pinfo["bbox"] = bbox
                # Arm theft watch once package has rested stationary for >= 4.0 seconds
                if (timestamp - pinfo["first_seen"]) >= 4.0:
                    pinfo["armed"] = True
            else:
                pid = self._pkg_seq
                self._pkg_seq += 1
                self.watched_packages[pid] = {
                    "label": lbl,
                    "center": (cx, cy),
                    "bbox": bbox,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "armed": False,
                    "alerted": False
                }
                matched_pkg_ids.add(pid)

        # 3. Package Removal / Theft Detection
        # Trigger if an armed package disappears for > 1.2s while an Unknown person is present (or left <= 3s ago)
        # and no registered family member is present to pick it up.
        to_remove = []
        for pid, pinfo in self.watched_packages.items():
            if pid in matched_pkg_ids:
                continue

            missing_duration = timestamp - pinfo["last_seen"]
            if missing_duration < 1.2:
                continue

            recent_unknown = bool(unknown_present_ids) or ((timestamp - self.last_unknown_seen_time) <= 3.0)
            if pinfo["armed"] and not pinfo["alerted"] and recent_unknown and not known_present:
                pinfo["alerted"] = True
                suspect_id = unknown_present_ids[0] if unknown_present_ids else self.last_unknown_id
                alert_frame = annotated_frame.copy()
                bx1, by1, bx2, by2 = pinfo["bbox"]
                cv2.rectangle(alert_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(
                    alert_frame,
                    f"MISSING PACKAGE ({pinfo['label'].upper()})",
                    (bx1, max(20, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )
                snap_name, snap_path = self._save_snapshot(
                    alert_frame, snapshot_dir, video_basename, f"package_theft_{pinfo['label']}", timestamp
                )
                new_events.append({
                    "event_type": "Package Removed",
                    "severity": "Critical",
                    "description": f"Stationary {pinfo['label']} was removed from doorway while unrecognized person (ID #{suspect_id}) was present!",
                    "snapshot_filename": snap_name,
                    "snapshot_filepath": snap_path,
                    "video_timestamp": timestamp
                })
                to_remove.append(pid)
            elif missing_duration > 4.0:
                to_remove.append(pid)

        for pid in to_remove:
            self.watched_packages.pop(pid, None)

        self.generated_events.extend(new_events)
        return new_events