from tracker.manualROI_CSRT import ManualROI_CSRTTracker
import os

# Paths
video_path = os.path.join("videos", "skydiver.mp4")
output_path = os.path.join("output", "skydiver_tracked_manualroi_csrt.mp4")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Initialize tracker
tracker = ManualROI_CSRTTracker(
    video_path=video_path,
    output_path=output_path
)

# Run processing
metrics = tracker.process()

# Print metrics safely
print("\n=== METRICS ===")
if metrics is not None:
    for k, v in metrics.items():
        if k != 'box_centers':
            print(f"{k}: {v}")
else:
    print("No metrics returned (check video input).")