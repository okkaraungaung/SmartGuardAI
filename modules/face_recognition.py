import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, yunet_path, sface_path, similarity_threshold=0.38, min_face_size=26):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.60,
            nms_threshold=0.3,
            top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config=""
        )

    def extract_embedding(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            return None

        h, w = bgr_image.shape[:2]
        if w < self.min_face_size or h < self.min_face_size:
            return None

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_image)
        if faces is None or len(faces) == 0:
            return None

        best_face = max(faces, key=lambda f: f[2] * f[3])
        if best_face[2] < self.min_face_size or best_face[3] < self.min_face_size:
            return None

        aligned_face = self.recognizer.alignCrop(bgr_image, best_face)
        feature = self.recognizer.feature(aligned_face)
        return feature[0].tolist()

    def analyze_face_pose(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            return {"detected": False, "too_far": False, "yaw": 0.0, "pitch": 0.0, "embedding": None}

        h, w = bgr_image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_image)

        if faces is None or len(faces) == 0:
            return {"detected": False, "too_far": False, "yaw": 0.0, "pitch": 0.0, "embedding": None}

        best_face = max(faces, key=lambda f: f[2] * f[3])
        if best_face[2] < 55 or best_face[3] < 55:
            return {"detected": True, "too_far": True, "yaw": 0.0, "pitch": 0.0, "embedding": None}

        rx, ry = best_face[4], best_face[5]
        lx, ly = best_face[6], best_face[7]
        nx, ny = best_face[8], best_face[9]
        rmy = best_face[11]
        lmy = best_face[13]

        eye_mid_x = (rx + lx) / 2.0
        eye_mid_y = (ry + ly) / 2.0
        mouth_mid_y = (rmy + lmy) / 2.0

        eye_dist = max(abs(lx - rx), 1.0)
        eye_to_mouth = max(abs(mouth_mid_y - eye_mid_y), 1.0)

        yaw = float((nx - eye_mid_x) / eye_dist)
        pitch = float((ny - eye_mid_y) / eye_to_mouth)

        aligned_face = self.recognizer.alignCrop(bgr_image, best_face)
        feature = self.recognizer.feature(aligned_face)

        return {
            "detected": True,
            "too_far": False,
            "yaw": round(yaw, 4),
            "pitch": round(pitch, 4),
            "embedding": feature[0].tolist()
        }

    def match_embedding(self, embedding, registered_users):
        if not registered_users or embedding is None:
            return "Unknown", 0.0

        query_vec = np.array([embedding], dtype=np.float32)
        user_scores = []

        for user in registered_users:
            emb_list = user["embedding"]
            if emb_list and isinstance(emb_list[0], (int, float)):
                emb_list = [emb_list]

            mat = np.array(emb_list, dtype=np.float32)
            centroid = np.mean(mat, axis=0, keepdims=True)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            candidates = [centroid] + [np.array([e], dtype=np.float32) for e in emb_list]

            max_u_score = 0.0
            for ref_vec in candidates:
                score = float(self.recognizer.match(
                    query_vec,
                    ref_vec,
                    cv2.FaceRecognizerSF_FR_COSINE
                ))
                if score > max_u_score:
                    max_u_score = score

            user_scores.append((user["name"], max_u_score))

        user_scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = user_scores[0]

        if best_score < self.similarity_threshold:
            return "Unknown", round(best_score, 2)

        # If multiple users are registered, require at least 0.04 separation from runner-up
        if len(user_scores) > 1 and (best_score - user_scores[1][1]) < 0.04:
            return "Unknown", round(best_score, 2)

        return best_name, round(best_score, 2)

    def identify_faces_in_frame(self, sharp_frame, person_detections, registered_users):
        results_map = {}
        if not person_detections:
            return results_map

        h, w = sharp_frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(sharp_frame)

        if faces is None or len(faces) == 0:
            return results_map

        for face in faces:
            fx, fy, fw, fh = face[0], face[1], face[2], face[3]
            if fw < self.min_face_size or fh < self.min_face_size:
                continue

            face_cx = fx + fw / 2.0
            face_cy = fy + fh / 2.0

            matched_idx = None
            for idx, det in enumerate(person_detections):
                if det["label"] != "person":
                    continue
                x1, y1, x2, y2 = det["bbox"]
                head_bottom = y1 + (y2 - y1) * 0.65
                if (x1 - 15) <= face_cx <= (x2 + 15) and (y1 - 15) <= face_cy <= head_bottom:
                    matched_idx = idx
                    break

            if matched_idx is None:
                continue

            if not registered_users:
                results_map[matched_idx] = ("Unknown", 0.0)
            else:
                aligned = self.recognizer.alignCrop(sharp_frame, face)
                feat = self.recognizer.feature(aligned)[0].tolist()
                name, score = self.match_embedding(feat, registered_users)
                results_map[matched_idx] = (name, score)

        return results_map