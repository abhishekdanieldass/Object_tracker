import cv2
import numpy as np
import time


class MotionStabilizedTracker:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path

        self.trail_points = []

        self.metrics = {
            "total_frames": 0,
            "tracked_frames": 0,
            "track_loss_count": 0,
            "fps_list": [],
            "box_centers": [],
        }

        self.min_area = 200
        self.scene_cut_threshold = 0.4

    # ---------------------------------------------------
    # 1. Scene Cut Detection
    # ---------------------------------------------------
    def detect_scene_cut(self, prev_gray, curr_gray):
        hist1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return score > self.scene_cut_threshold

    # ---------------------------------------------------
    # 2. Estimate Camera Motion
    # ---------------------------------------------------
    def estimate_global_motion(self, prev_gray, curr_gray):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 10)

        if prev_pts is None:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 10:
            return None

        M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        return M

    # ---------------------------------------------------
    # 3. Frame Differencing (with stabilization)
    # ---------------------------------------------------
    def compute_frame_diff(self, prev_gray, curr_gray, width, height):
        M = self.estimate_global_motion(prev_gray, curr_gray)

        if M is not None:
            stabilized = cv2.warpAffine(curr_gray, M, (width, height))
        else:
            stabilized = curr_gray

        diff = cv2.absdiff(prev_gray, stabilized)
        return diff

    # ---------------------------------------------------
    # 4. Adaptive Threshold (stronger)
    # ---------------------------------------------------
    def adaptive_threshold(self, diff):
        mean = np.mean(diff)
        std = np.std(diff)

        thresh_val = mean + 2.5 * std  # 🔥 stronger threshold

        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        return thresh

    # ---------------------------------------------------
    # 5. Morphological Cleanup (refined)
    # ---------------------------------------------------
    def clean_mask(self, thresh):
        kernel = np.ones((3, 3), np.uint8)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        return thresh

    # ---------------------------------------------------
    # 6. Contour Detection (with filtering)
    # ---------------------------------------------------
    def detect_object(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Sort contours by size
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        frame_h, frame_w = mask.shape
        frame_area = frame_h * frame_w

        for c in contours[:3]:  # check top 3
            area = cv2.contourArea(c)

            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            bbox_area = w * h

            # ❌ Reject large regions (background)
            if bbox_area > 0.25 * frame_area:
                continue

            # ❌ Reject overly wide/tall boxes
            if w > 0.5 * frame_w or h > 0.5 * frame_h:
                continue

            return x, y, w, h

        return None

    # ---------------------------------------------------
    # 7. Draw Tracking
    # ---------------------------------------------------
    def draw_tracking(self, frame, bbox):
        x, y, w, h = bbox

        cx = x + w // 2
        cy = y + h // 2

        self.trail_points.append((cx, cy))
        self.metrics["box_centers"].append((cx, cy))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
        cv2.putText(frame, "Object", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        for i in range(1, len(self.trail_points)):
            cv2.line(frame,
                     self.trail_points[i - 1],
                     self.trail_points[i],
                     (0, 255, 255), 2)

    # ---------------------------------------------------
    # 8. Main Loop
    # ---------------------------------------------------
    def process(self):
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        ret, prev_frame = cap.read()
        if not ret:
            print("Error reading video")
            return {}

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.metrics["total_frames"] += 1
            start_time = time.time()

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Scene cut handling
            if self.detect_scene_cut(prev_gray, curr_gray):
                print(f"Scene cut at frame {frame_idx}")

                self.trail_points = []
                prev_gray = curr_gray
                self.metrics["track_loss_count"] += 1
                out.write(frame)
                frame_idx += 1
                continue

            # Motion pipeline
            diff = self.compute_frame_diff(prev_gray, curr_gray, width, height)
            thresh = self.adaptive_threshold(diff)
            mask = self.clean_mask(thresh)

            bbox = self.detect_object(mask)

            detected = False

            if bbox:
                self.draw_tracking(frame, bbox)
                self.metrics["tracked_frames"] += 1
                detected = True
            else:
                self.metrics["track_loss_count"] += 1

            # FPS
            elapsed = time.time() - start_time
            fps_val = 1.0 / elapsed if elapsed > 0 else 0
            self.metrics["fps_list"].append(fps_val)

            # Overlay
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame,
                        f"Status: {'Tracking' if detected else 'Lost'}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if detected else (0, 0, 255),
                        2)

            out.write(frame)

            prev_gray = curr_gray
            frame_idx += 1

        cap.release()
        out.release()

        return self.compute_metrics()

    # ---------------------------------------------------
    # 9. Metrics
    # ---------------------------------------------------
    def compute_metrics(self):
        total = self.metrics["total_frames"]
        tracked = self.metrics["tracked_frames"]

        return {
            "total_frames": total,
            "tracked_frames": tracked,
            "track_loss_count": self.metrics["track_loss_count"],
            "tracking_rate": round((tracked / total) * 100, 2)
            if total > 0 else 0,
            "avg_fps": round(np.mean(self.metrics["fps_list"]), 2),
        }