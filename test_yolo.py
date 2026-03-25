from tracker.yolo_tracker import YOLOTracker

tracker = YOLOTracker(
    video_path= "videos\skydiver.mp4",
    output_path= "output\skydiver_tracked_yolo.mp4",
    show_bbox=True,
    show_trail= True
)

metrics = tracker.process()
print("\n=== METRICS ===")
for k, v in metrics.items():
    if k != 'box_centers':
        print(f"{k}: {v}")