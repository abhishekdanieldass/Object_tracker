import cv2
import numpy as np
import time

class CSRTTracker:
    """
    Object-agnostic detect-and-track pipeline.
    
    Detection: Temporal frame differencing
               with multi-criterion candidate scoring
    Tracking:  CSRT between detections
    Re-init:   Automatic on track loss
    """
    
    def __init__(self, video_path, output_path,
                 detect_interval=30,
                 aggregation_window=5):
        
        self.video_path    = video_path
        self.output_path   = output_path
        self.detect_interval    = detect_interval
        self.aggregation_window = aggregation_window
        
        self.tracker             = None
        self.tracker_initialized = False
        self.last_bbox           = None
        
        self.metrics = {
            'model_name':       'CSRT Tracker',
            'total_frames':     0,
            'tracked_frames':   0,
            'track_loss_count': 0,
            'fps_list':         [],
            'box_centers':      [],
            'box_areas':        [],
            'detection_count':  0,
        }

    # ── DETECTION ─────────────────────────────────────────

    def _score_candidate(self, contour, diff_aggregated,
                         frame_area):
        """
        Score a contour as object candidate.
        Higher = more likely to be our object.
        """
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Motion score: avg pixel change in this region
        roi = diff_aggregated[y:y+h, x:x+w]
        motion_score = float(np.mean(roi)) if roi.size > 0 else 0
        
        # Size score: prefer small-medium objects
        size_ratio = area / frame_area
        # Gaussian peak centered at 0.5% of frame
        size_score = np.exp(
            -((size_ratio - 0.005) ** 2) / (2 * 0.003**2))
        
        # Compactness: real objects tend to be blob-like
        perimeter = cv2.arcLength(contour, True)
        compactness = (4 * np.pi * area) / (
            perimeter ** 2 + 1e-5)
        
        # Weighted combination
        score = (
            0.5 * motion_score +
            0.3 * size_score * 100 +
            0.2 * compactness * 100
        )
        return score

    def _detect_object(self, frame_buffer, current_frame):
        """
        Detect moving object using temporal aggregation
        + multi-criterion scoring.
        """
        if len(frame_buffer) < 2:
            return None
        
        height, width = frame_buffer[0].shape
        frame_area = height * width
        
        # Step 1: Frame differences
        diffs = []
        for i in range(1, len(frame_buffer)):
            diff = cv2.absdiff(
                frame_buffer[i-1], frame_buffer[i])
            diffs.append(diff.astype(np.float32))
        
        # Step 2: Temporal aggregation
        # Mean suppresses single-frame noise
        aggregated = np.mean(diffs, axis=0).astype(np.uint8)
        
        # Step 3: Threshold
        # OTSU finds optimal threshold automatically
        _, binary = cv2.threshold(
            aggregated, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 4: Morphological cleanup
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        # Open: remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3)
        # Close: fill gaps in object
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5)
        # Dilate: expand slightly for better CSRT init
        binary = cv2.dilate(binary, k3, iterations=1)
        
        # Step 5: Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Step 6: Filter by size
        min_area = frame_area * 0.0003   # 0.03% of frame
        max_area = frame_area * 0.4      # 40% of frame
        
        valid = [
            c for c in contours
            if min_area < cv2.contourArea(c) < max_area
        ]
        
        if not valid:
            return None
        
        # Step 7: Score and pick best candidate
        scored = [
            (c, self._score_candidate(c, aggregated, frame_area))
            for c in valid
        ]
        best_contour = max(scored, key=lambda x: x[1])[0]
        
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(width  - x, w + 2 * pad)
        h = min(height - y, h + 2 * pad)
        
        return (x, y, w, h)

    # ── TRACKER ───────────────────────────────────────────

    def _init_tracker(self, frame, bbox):
        """Initialize fresh CSRT instance"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.tracker_initialized = True
        self.last_bbox = bbox
        self.metrics['detection_count'] += 1

    def _update_tracker(self, frame):
        """Update CSRT, return (success, bbox)"""
        if not self.tracker_initialized:
            return False, None
        success, bbox = self.tracker.update(frame)
        if success:
            self.last_bbox = bbox
            return True, bbox
        self.tracker_initialized = False
        return False, None

    # ── MAIN LOOP ─────────────────────────────────────────

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path, fourcc, fps, (width, height))
        
        frame_buffer = []   # rolling grayscale window
        frame_idx    = 0
        
        print(f"Processing... detect every {self.detect_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.metrics['total_frames'] += 1
            t0 = time.time()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Rolling buffer
            frame_buffer.append(gray.copy())
            if len(frame_buffer) > self.aggregation_window:
                frame_buffer.pop(0)
            
            detected  = False
            bbox      = None
            
            # ── DECIDE WHETHER TO DETECT ──────────────────
            should_detect = (
                frame_idx % self.detect_interval == 0 or
                not self.tracker_initialized
            )
            
            if (should_detect and
                    len(frame_buffer) >= self.aggregation_window):
                
                bbox = self._detect_object(
                    frame_buffer, frame)
                
                if bbox is not None:
                    self._init_tracker(frame, bbox)
            
            # ── CSRT UPDATE ───────────────────────────────
            success, bbox = self._update_tracker(frame)
            
            if success and bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                cx   = x + w // 2
                cy   = y + h // 2
                area = w * h
                
                self.metrics['box_centers'].append((cx, cy))
                self.metrics['box_areas'].append(area)
                self.metrics['tracked_frames'] += 1
                detected = True
                
                # Clean bounding box only — no trail
                cv2.rectangle(
                    frame, (x, y), (x+w, y+h),
                    (0, 255, 0), 2)
                
                # Label with mode
                mode_label = (
                    "RE-DETECTED" if should_detect
                    else "TRACKING")
                cv2.putText(
                    frame, mode_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255) if should_detect
                    else (0, 255, 0),
                    2)
            
            if not detected:
                self.metrics['track_loss_count'] += 1
                # Show loss indicator
                cv2.putText(
                    frame, "SEARCHING...",
                    (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
            
            # ── INFO OVERLAY ──────────────────────────────
            elapsed    = time.time() - t0
            frame_fps  = 1.0 / elapsed if elapsed > 0 else 0
            self.metrics['fps_list'].append(frame_fps)
            
            cv2.putText(frame,
                       "Model: BG Sub + CSRT",
                       (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.55, (255, 255, 0), 2)
            cv2.putText(frame,
                       f"Frame: {frame_idx} | "
                       f"FPS: {frame_fps:.1f}",
                       (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.55, (255, 255, 255), 2)
            cv2.putText(frame,
                       f"Detections: "
                       f"{self.metrics['detection_count']}",
                       (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.55, (255, 255, 255), 2)
            
            if frame_idx % 100 == 0:
                print(f"Frame {frame_idx:4d} | "
                      f"Detected: {detected} | "
                      f"FPS: {frame_fps:.1f} | "
                      f"Reinits: "
                      f"{self.metrics['detection_count']}")
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print("\nDone.")
        return self._compute_final_metrics()

    def _compute_final_metrics(self):
        total   = self.metrics['total_frames']
        tracked = self.metrics['tracked_frames']
        return {
            'model_name':       'Background Sub + CSRT',
            'total_frames':     total,
            'tracked_frames':   tracked,
            'track_loss_count': self.metrics['track_loss_count'],
            'tracking_rate':    round(
                tracked/total*100, 2) if total > 0 else 0,
            'avg_fps':          round(
                np.mean(self.metrics['fps_list']), 2),
            'avg_box_area':     round(
                np.mean(self.metrics['box_areas']), 2
            ) if self.metrics['box_areas'] else 0,
            'box_centers':      self.metrics['box_centers'],
            'detection_count':  self.metrics['detection_count'],
        }