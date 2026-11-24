import os
import cv2


class SecurityEventEngine:
    def __init__(self, loitering_threshold=15.0, cooldown_seconds=60.0, enable_unknown=True, enable_loitering=True, enable_package=True, enable_vehicle=False):
        self.loitering_threshold = float(loitering_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.enable_unknown = enable_unknown
        self.enable_loitering = enable_loitering
        self.enable_package = enable_package

        self.alert_history = {}
        self.generated_events = []

    def _can_trigger(self, key, timestamp):
        last_time = self.alert_history.get(key)
        if last_time is None or (timestamp - last_time) >= self.cooldown_seconds:
            self.alert_history[key] = timestamp
            return True
        return False

    def _save_snapshot(self, annotated_frame, snapshot_dir, video_basename, event_slug, timestamp):
        filename = f"{video_basename}_{event_slug}_{int(timestamp*10)}.jpg"
        filepath = os.path.join(snapshot_dir, filename)
        cv2.imwrite(filepath, annotated_frame)
        return filename, filepath

    def evaluate_frame(self, detections, tracker_mgr, annotated_frame, snapshot_dir, video_basename, timestamp):
        new_events = []

        # 1. Alert ONLY when an Unknown (unregistered) person stays longer than loitering_threshold (15s)
        for tid, info in tracker_mgr.tracks.items():
            if info["last_seen"] != timestamp:
                continue

            dwell_time = round(max(0.0, info["last_seen"] - info["first_seen"]), 1)
            name = info["name"]

            # Ignore registered family members and tracks still scanning
            if name != "Unknown":
                continue

            if (self.enable_unknown or self.enable_loitering) and dwell_time > self.loitering_threshold:
                key = f"unknown_loitering_{tid}"
                if self._can_trigger(key, timestamp):
                    snap_name, snap_path = self._save_snapshot(
                        annotated_frame, snapshot_dir, video_basename, f"unknown_loiter_id{tid}", timestamp
                    )
                    new_events.append({
                        "event_type": "Unknown Loitering",
                        "severity": "Critical",
                        "description": f"Unrecognized person (Track ID #{tid}) stayed at entrance for {dwell_time}s (limit: {self.loitering_threshold}s).",
                        "snapshot_filename": snap_name,
                        "snapshot_filepath": snap_path,
                        "video_timestamp": timestamp
                    })

        # 2. Package-Like Objects
        for det in detections:
            lbl = det["label"]
            tid = det.get("track_id") or 0

            if self.enable_package and lbl in {"backpack", "handbag", "suitcase"}:
                key = f"package_{lbl}_{tid}"
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

        self.generated_events.extend(new_events)
        return new_events