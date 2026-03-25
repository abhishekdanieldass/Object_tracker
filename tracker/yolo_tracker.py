import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLOTracker:
    def __init__(self, video_path, output_path, show_bbox = True, show_trail = True):
        self.video_path = video_path
        self.output_path = output_path
        self.show_bbox = show_bbox
        self.show_trail = show_trail
        
        # self.model = YOLO("yolov8n.pt")
        self.model = YOLO("yolo11s.pt")

        self.trail_points = []

        self.metrics = {
            "model_name": "YOLOv11",
            "total_frames": 0,
            "tracked_frames": 0,
            "track_loss_count": 0,
            "fps_list": [],
            "box_centers": [],
            "box_areas": []
        }
    
    def process(self):
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_idx = 0

        while True:
            ret,frame = cap.read()
            if not ret:
                break

            self.metrics['total_frames'] += 1
            start_time = time.time()

            results = self.model.track(
                frame,
                persist = True,
                classes = [0],
                conf = 0.3,
                verbose = False
            )

            inference_time =  time.time() - start_time
            frame_fps = 1.0 / inference_time if inference_time > 0 else 0
            self.metrics['fps_list'].append(frame_fps)

            detected = False

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confs = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confs)

                box = boxes.xyxy[best_idx].cpu().numpy()
                conf = float(confs[best_idx])

                x1,y1,x2,y2 = map(int,box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2- y1)

                self.trail_points.append((cx,cy))
                self.metrics['box_centers'].append((cx,cy))
                self.metrics['box_areas'].append(area)
                self.metrics['tracked_frames'] += 1
                detected = True

                if self.show_bbox:
                    cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Skydiver {conf: .2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not detected:
                self.metrics['track_loss_count'] += 1

            if self.show_trail and len(self.trail_points) > 1:
                for i in range(1, len(self.trail_points)):
                    alpha = i / len(self.trail_points)
                    color = (
                        int(255 * alpha),
                        int(100* alpha),
                        255
                    )

                    cv2.line(
                        frame,
                        self.trail_points[i-1],
                        self.trail_points[i],
                        color,
                        2
                    )

            cv2.putText(frame, f"Frame: {frame_idx}", (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Model: YOLOv8", (10,55), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255,255,0), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return self._compute_final_metrics()
    
    def _compute_final_metrics(self):
        total = self.metrics['total_frames']
        tracked = self.metrics['tracked_frames']

        return{
            "model_name": "YOLOv11",
            "total_frame": total,
            "tracked_frames": tracked,
            "track_loss_count":self.metrics["track_loss_count"],
            "tracking_rate": round(tracked/total * 100, 2) if total > 0  else 0,
            "avg_fps": round(np.mean(self.metrics["fps_list"]), 2),
            "avg_box_area": round(np.mean(self.metrics["box_areas"]), 2) if self.metrics["box_areas"] else 0,
            "box_centers": self.metrics["box_centers"]
        }
    



                



