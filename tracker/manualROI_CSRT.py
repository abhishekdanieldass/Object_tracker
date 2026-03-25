import cv2
import time
import numpy as np


class ManualROI_CSRTTracker:

    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path

        # Metrics
        self.metrics = {
            'model_name': 'Manual ROI + CSRT',
            'total_frames': 0,
            'tracked_frames': 0,
            'track_loss_count': 0,
            'fps_list': [],
            'box_centers': [],
            'box_areas': [],
            'reinit_count': 0,
        }

    def open_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video")
            return None
        return cap

    def create_writer(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        return cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (width, height)
        )

    def create_tracker(self):
        return cv2.TrackerCSRT_create()

    def select_object(self, frame):
        bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
        return bbox

    def process(self):
        cap = self.open_video()
        if cap is None:
            return None

        out = self.create_writer(cap)

        ret, frame = cap.read()
        if not ret:
            return None

        # Initial selection
        tracker = self.create_tracker()
        bbox = self.select_object(frame)
        tracker.init(frame, bbox)
        self.metrics['reinit_count'] += 1

        frame_idx = 1
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.metrics['total_frames'] += 1

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            self.metrics['fps_list'].append(fps)

            frame_idx += 1

            success, bbox = tracker.update(frame)

            # 🔥 SHOW FRAME FIRST (so user can act anytime)
            display_frame = frame.copy()

            # Overlay basic info
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(display_frame, f"Frame: {frame_idx}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(display_frame, "Press R to reinitialize | Q to quit",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Draw tracking result
            if success:
                x, y, w, h = [int(v) for v in bbox]

                self.metrics['tracked_frames'] += 1

                cx = x + w // 2
                cy = y + h // 2
                area = w * h

                self.metrics['box_centers'].append((cx, cy))
                self.metrics['box_areas'].append(area)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, "Tracking", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                self.metrics['track_loss_count'] += 1

                cv2.putText(display_frame,
                            "TRACK LOST",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            # 🔥 SHOW FRAME
            cv2.imshow("Tracking Review", display_frame)

            key = cv2.waitKey(30) & 0xFF

            # 🔥 FORCE REINITIALIZATION ANYTIME
            if key == ord('r'):
                new_bbox = self.select_object(frame)
                tracker = self.create_tracker()
                tracker.init(frame, new_bbox)
                self.metrics['reinit_count'] += 1
                continue

            elif key == ord('q'):
                break

            # Write frame
            out.write(display_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("Done.")
        return self._compute_final_metrics()

    def _compute_final_metrics(self):
        total = self.metrics['total_frames']
        tracked = self.metrics['tracked_frames']

        return {
            'model_name': self.metrics['model_name'],
            'total_frames': total,
            'tracked_frames': tracked,
            'track_loss_count': self.metrics['track_loss_count'],
            'tracking_rate': round(tracked / total * 100, 2) if total > 0 else 0,
            'avg_fps': round(np.mean(self.metrics['fps_list']), 2) if self.metrics['fps_list'] else 0,
            'avg_box_area': round(np.mean(self.metrics['box_areas']), 2) if self.metrics['box_areas'] else 0,
            'reinit_count': self.metrics['reinit_count'],
        }