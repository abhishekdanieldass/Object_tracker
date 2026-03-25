# Aerial Object Tracking Pipeline

A multi-model computer vision pipeline for tracking small aerial objects in video footage. Demonstrated on BASE jump / wingsuit footage. Directly extensible to drone, UAV, and bird tracking.

---

## Problem Statement

Tracking small aerial objects in real-world footage is a challenging CV problem due to:
- Small object size relative to frame
- Non-uniform camera motion
- Low contrast between object and background
- Complex, dynamic backgrounds

---

## Models Implemented

| Model | Approach | Tracking Rate |
|-------|----------|--------------|
| YOLOv8 | Pretrained CNN detection | ~4% |
| Lucas-Kanade Optical Flow | Sparse feature tracking | ~20% |
| CSRT (Manual Init) | Discriminative correlation filter | 98.62% |

---

## Pipeline Architecture
```
Input Video
     ↓
Frame Extraction (OpenCV)
     ↓
┌─────────────────────────────────┐
│  Model Selection                │
│  ├── YOLOv8 (deep learning)     │
│  ├── Optical Flow (classical)   │
│  └── CSRT (hybrid)              │
└─────────────────────────────────┘
     ↓
Bounding Box + Tracking
     ↓
Metrics Evaluation
     ↓
Annotated Output Video
```

---

## Key Findings

**YOLOv8** failed on this scenario because the skydiver occupies less than 0.5% of the frame — below the reliable detection threshold for models trained on standard COCO data.

**Optical Flow** improved results but struggled with non-uniform camera motion. The dominant motion vector (camera pan) was difficult to separate from object motion reliably.

**CSRT with manual initialization** achieved 98.62% tracking rate. Manual ROI selection ensures correct object identification — a deliberate human-in-the-loop design decision prioritizing robustness over full automation.

---

## Why Not Fully Automated?

Full automation was attempted using:
- Background subtraction with temporal aggregation
- Multi-criterion candidate scoring
- Automatic CSRT re-initialization

These approaches struggled with the low signal-to-noise ratio of this footage. The honest engineering decision was to use semi-automatic tracking rather than ship a fully automated system with 20% accuracy.

---

## Extensibility

This pipeline is object-agnostic beyond the initialization step:
```
Skydiver → manual ROI → CSRT tracks appearance
Drone    → manual ROI → CSRT tracks appearance  
Bird     → manual ROI → CSRT tracks appearance
```

For full automation on drone/UAV detection:
- Fine-tune YOLOv8 on domain-specific aerial dataset
- Add DeepSORT for appearance-based re-identification
- Incorporate thermal/IR modality for low-contrast scenarios

---

## Setup
```bash
# Clone repo
git clone https://github.com/abhishekdanieldass/Object_tracker.git
cd Object_tracker

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Input video sourced from YouTube:
```
https://youtube.com/shorts/EPB4JXPo3nY
```

To download:
```bash
pip install yt-dlp
yt-dlp -o videos/skydiver.mp4 https://youtube.com/shorts/EPB4JXPo3nY
```

---

## Usage
```bash
# CSRT Tracker (best performance)
python tracker/csrt_tracker.py

# Optical Flow Tracker
python tracker/optical_flow_tracker.py

# YOLOv8 Tracker
python tracker/yolo_tracker.py
```

---

## Project Structure
```
Object_tracker/
├── tracker/
│   ├── csrt_tracker.py          # CSRT detect-and-track
│   ├── optical_flow_tracker.py  # LK sparse optical flow
│   └── yolo_tracker.py          # YOLOv8 detection
├── report/
│   └── Report_Abhishek_Daniel.pdf
├── requirements.txt
└── README.md
```

---

## Limitations

- Manual initialization required for reliable tracking
- RGB-only pipeline struggles with low contrast
- Performance degrades on abrupt scene transitions

## Future Work

- Thermal/infrared modality for improved object visibility
- Fine-tuned small object detection model
- DeepSORT integration for appearance re-identification
- Hybrid detect-and-track with automatic initialization

---

## Author

**Abhishek Daniel Dass**
abhishekdaniel1411@gmail.com
