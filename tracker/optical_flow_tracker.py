import cv2
import numpy as np
import time

class OpticalFlowTracker:
    """
    Tracks objects in video using Lucas-Kanade Sparse Optical Flow.

    Approach:
    1. Detect feature points (using Shi-Tomasi corners)
    2. Track points frame to frame (LK optical flow)
    3. Estimate camera motion (median of all vectors)
    4. Subtract camera motion to find the object motion
    5. Cluster object points into a bounding box
    """

    def __init__(self, video_path, output_path, show_bbox = True, show_trail = True):
        self.video_path = video_path
        self.output_path = output_path
        self.show_bbox = show_bbox
        self.show_trail = show_trail

        self.lk_params = dict(
            winSize = (21,21),
            maxLevel = 3,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30, 0.01
            )
        )

        self.feature_params = dict(
            maxCorners = 300,
            qualityLevel = 0.01,
            minDistance = 5,
            blockSize = 7
        )

        self.trail_points = []
        self.metrics = {
            "model_name": "Optical FLow (LK)",
            "total_frames": 0,
            "tracked_frames": 0,
            "track_loss_count": 0,
            "fps_list": [],
            "box_centers": [],
            "box_areas": []
        }

    def _get_dominant_motion(self, old_pts, new_pts, status):
        """
        Estimate camera motion using median of all tracked point displacements.

        why median not mean?
        Mean is sensitive to outliers (object points) so using Median gives us true background motion.
        """
        good_old = old_pts[status == 1]
        good_new = new_pts[status == 1]

        if len(good_old) < 2:
            return np.array([0.0, 0.0])
        
        motions = good_new - good_old
        dominant = np.median(motions, axis=0)
        return dominant
    
    def _find_object_points(self, old_pts, new_pts, status, dominant_motion):
        """
        After removing camera motion, find points that are moving independently aka our object of interest.
        """
        good_old = old_pts[status == 1]
        good_new = new_pts[status == 1]

        if len(good_old) < 2:
            return None, None
        
        motions = good_new - good_old
        relative_motion = motions - dominant_motion
        magnitudes = np.linalg.norm(relative_motion, axis=1)
        threshold = np.percentile(magnitudes, 80)

        if threshold < 0.5:
            return None, None
        
        object_mask = magnitudes > threshold
        object_points = good_new[object_mask]

        if len(object_points) < 3:
            return None, None
        
        return object_points, good_new[~object_mask]
    
    def _points_to_bbox(self, points, frame_w, frame_h, margin = 15):
        """
        Convert cluster of object points to bounding box.
        Margin adds padding arounf the tight box.
        This is to clamp the boundaries because cv2.rectangle with negative values crashes the program.
        """
        x_min = int(np.min(points[:, 0])) - margin
        y_min = int(np.min(points[:, 1])) - margin
        x_max = int(np.min(points[:, 0])) + margin
        y_max = int(np.min(points[:, 1])) + margin

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame_w, x_max)
        y_max = min(frame_h, y_max)

        return x_min, y_min, x_max, y_max
    
    def process(self):
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        ret, old_frame = cap.read()
        if not ret:
            print("Error: Could not read video")
            return {}
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        trail_mask = np.zeros_like(old_frame)

        frame_idx = 0

        print(f"Processing {self.video_path}")
        print(f"Resolution: {width} x {height} @ {fps} fps")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.metrics["total_frames"] += 1
            start_time = time.time()

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = False

            if p0 is not None and len(p0) > 0:
                p1, status, err = cv2.calcOpticalFlowPyrLK(
                    old_gray,
                    frame_gray,
                    p0,
                    None,
                    **self.lk_params
                )

                if p1 is not None and status is not None:
                    st = status.ravel()

                    dominant = self._get_dominant_motion(
                        p0.reshape(-1,2),
                        p1.reshape(-1,2),
                        st
                    )

                    obj_pts, bg_pts = self._find_object_points(
                        p0.reshape(-1,2),
                        p1.reshape(-1,2),
                        st,
                        dominant
                    )

                    if obj_pts is not None:
                        x1,y1,x2,y2 = self._points_to_bbox(obj_pts, width, height)

                        cx = (x1+x2)//2
                        cy = (y1+y2)//2
                        area = (x2-x1) * (y2-y1)

                        self.trail_points.append((cx,cy))
                        self.metrics["box_centers"].append((cx,cy))
                        self.metrics["box_areas"].append(area)
                        self.metrics["tracked_frames"] += 1
                        detected =True

                        if self.show_bbox:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 165,0), 2)
                            cv2.putText(frame, "Object", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
                            for pt in obj_pts:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
                    
                    p0 = p1[st == 1].reshape(-1,1,2)
                
                if self.show_trail and len(self.trail_points) > 1:
                    for i in range(1, len(self.trail_points)):
                        alpha = i / len(self.trail_points)
                        color = (int(255*alpha), int(165*alpha), 0)
                        cv2.line(
                            trail_mask, 
                            self.trail_points[i-1],
                            self.trail_points[i],
                            color,
                            2
                        )
                
                if (frame_idx % 30 == 0 or p0 is None or len(p0) < 10):
                    p0 = cv2.goodFeaturesToTrack(
                        frame_gray,
                        mask=None,
                        **self.feature_params
                    )

                if not detected:
                    self.metrics["track_loss_count"] += 1

                output = cv2.add(frame,trail_mask)
                
                elapsed = time.time() - start_time
                frame_fps = 1.0/elapsed if elapsed > 0 else 0
                self.metrics["fps_list"].append(frame_fps)
                cv2.putText(output, "Model: Optical Flow (LK)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(output, f"Frame: {frame_idx}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(output, f"FPS: {frame_fps: .1f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(output, f"Status: {'tracking' if detected else 'Lost' }", (10,105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if detected else (0,0,255), 2)

                if frame_idx % 100 == 0:
                    print(f"Frame {frame_idx}/1016 | "
                          f"Tracked: {detected} | "
                          f"FPS: {frame_fps: .1f}")
                
                out.write(output)

                old_gray = frame_gray.copy()
                frame_idx += 1

        cap.release()
        out.release()
        print(f"Done. Output: {self.output_path}")
        return self._compute_final_metrics()
    
    def _compute_final_metrics(self):
        """
        Aggregate per-frame metrics into
        summary statistics for comparison
        """
        total   = self.metrics['total_frames']
        tracked = self.metrics['tracked_frames']
        
        return {
            'model_name': 'Optical Flow (LK)',
            'total_frames': total,
            'tracked_frames': tracked,
            'track_loss_count': self.metrics['track_loss_count'],
            'tracking_rate': round(
                tracked / total * 100, 2) if total > 0 else 0,
            'avg_fps': round(
                np.mean(self.metrics['fps_list']), 2),
            'avg_box_area': round(
                np.mean(self.metrics['box_areas']), 2
            ) if self.metrics['box_areas'] else 0,
            'box_centers': self.metrics['box_centers'],
        }







