from collections import deque, Counter


class PersonTrackerManager:
    def __init__(self):
        self.tracks = {}

    def update(self, track_id, timestamp, bbox, identity="NoFace", face_score=0.0):
        if track_id is None:
            return "Scanning..." if identity == "NoFace" else identity

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "track_id": track_id,
                "name": "Scanning...",
                "votes": deque(maxlen=7),
                "best_face_score": 0.0,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "frame_hits": 1,
                "last_bbox": bbox
            }

        record = self.tracks[track_id]
        record["last_seen"] = timestamp
        record["frame_hits"] += 1
        record["last_bbox"] = bbox

        if identity != "NoFace":
            record["votes"].append(identity)
            if identity != "Unknown" and face_score > record["best_face_score"]:
                record["best_face_score"] = face_score

        # Require multi-frame consensus (at least 2 consistent matches) to avoid false names
        if len(record["votes"]) > 0:
            counts = Counter(record["votes"])
            top_name, top_count = counts.most_common(1)[0]
            if top_name != "Unknown" and top_count >= 2:
                record["name"] = top_name
            elif top_name == "Unknown" and top_count >= 3:
                record["name"] = "Unknown"
        elif record["frame_hits"] >= 6 and record["name"] == "Scanning...":
            record["name"] = "Unknown"

        return record["name"]

    def get_active_summary(self, current_time, active_window=2.0):
        """Returns only visitors currently in frame (seen within the last 2 seconds)."""
        summary = []
        for tid, info in sorted(self.tracks.items()):
            if (current_time - info["last_seen"]) > active_window:
                continue
            dwell = round(max(0.0, info["last_seen"] - info["first_seen"]), 2)
            display_name = "Unknown" if info["name"] == "Scanning..." else info["name"]
            summary.append({
                "track_id": tid,
                "name": display_name,
                "first_seen": info["first_seen"],
                "last_seen": info["last_seen"],
                "dwell_time": dwell,
                "frame_hits": info["frame_hits"]
            })
        return summary

    def get_summary(self):
        summary = []
        for tid, info in sorted(self.tracks.items()):
            dwell = round(max(0.0, info["last_seen"] - info["first_seen"]), 2)
            display_name = "Unknown" if info["name"] == "Scanning..." else info["name"]
            summary.append({
                "track_id": tid,
                "name": display_name,
                "first_seen": info["first_seen"],
                "last_seen": info["last_seen"],
                "dwell_time": dwell,
                "frame_hits": info["frame_hits"]
            })
        return summary