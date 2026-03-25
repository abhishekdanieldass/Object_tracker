from tracker.optical_flow_tracker import OpticalFlowTracker

tracker = OpticalFlowTracker(
    video_path=r"videos\skydiver.mp4",
    output_path=r"output\tracked_optical.mp4",
    show_bbox=True,
    show_trail=True
)

metrics = tracker.process()
print("\n=== METRICS ===")
for k, v in metrics.items():
    if k != 'box_centers':
        print(f"{k}: {v}")