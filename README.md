# Data Science Final — Road Object Detection Dataset Preparation

## Project Goal
Multi-class object detection model (YOLOv8) for road scenes:
potholes, vehicles (by type), road line types, people, pedestrian lanes,
rumble strips, stop lights, crossing cones.

---

## Directory Structure

```
Data_Sci_Final/
│
├── Raw_Data/                    # Original videos (DO NOT MODIFY)
│   ├── VID_1.mov
│   ├── VID_2.MOV
│   ├── VID_3.MOV
│   └── VID_4.MOV
│
├── scripts/                     # Data preparation pipeline
│   ├── 01_extract_frames.py     # Step 1: Extract frames from videos
│   ├── 02_filter_blurry.py      # Step 2: Remove blurry/dark frames
│   ├── 03_split_dataset.py      # Step 3: Split into train/val/test
│   └── 04_class_stats.py        # Step 4: Check class distribution
│
├── extracted_frames/            # [AUTO] Raw extracted frames
│   ├── VID_1/
│   ├── VID_2/
│   ├── VID_3/
│   └── VID_4/
│
├── clean_frames/                # [AUTO] Frames after quality filter
│   ├── VID_1/
│   ├── VID_2/
│   ├── VID_3/
│   └── VID_4/
│
├── rejected_frames/             # [AUTO] Blurry/dark frames (for review)
│
├── dataset/                     # FINAL YOLO-ready dataset
│   ├── dataset.yaml             # YOLO config (class names + paths)
│   ├── images/
│   │   ├── train/               # 70% of images
│   │   ├── val/                 # 20% of images
│   │   └── test/                # 10% of images
│   └── labels/
│       ├── train/               # YOLO .txt labels (after annotation)
│       ├── val/
│       └── test/
│
└── README.md                    # This file
```

---

## How to Run (Step by Step)

### Prerequisites
```bash
pip install opencv-python-headless
```

### Step 1 — Extract Frames
Extracts 1 frame per second from VID_1–VID_4, resizes to 640×640, saves as JPEG.
```bash
python scripts/01_extract_frames.py
```
**Output:** `extracted_frames/VID_1/`, `VID_2/`, etc.

### Step 2 — Filter Blurry/Dark Frames
Removes low-quality frames automatically. Good frames go to `clean_frames/`.
```bash
python scripts/02_filter_blurry.py
```
**Tip:** Check `rejected_frames/` — if good images were rejected, lower
`BLUR_THRESHOLD` in the script (default: 50).

### Step 3 — Split into Train/Val/Test
Shuffles and splits clean frames into the YOLO folder structure.
```bash
python scripts/03_split_dataset.py
```
**Output:** `dataset/images/train/`, `val/`, `test/` with matching empty label files.

### Step 4 — Annotate (MANUAL STEP)
Label your images with bounding boxes. Options:
- **Roboflow** (recommended, web-based, fastest)
- **LabelImg** (local, free)
- **CVAT** (powerful, web-based)

Export in **YOLO format** and place `.txt` files in `dataset/labels/train/`, `val/`, `test/`.

### Step 5 — Check Class Distribution
After annotation, run this to find imbalanced classes:
```bash
python scripts/04_class_stats.py
```

---

## Class Mapping (13 Classes)

| ID | Class Name              |
|----|-------------------------|
| 0  | pothole                 |
| 1  | car                     |
| 2  | truck                   |
| 3  | motorcycle              |
| 4  | bus                     |
| 5  | person                  |
| 6  | traffic_light           |
| 7  | cone                    |
| 8  | pedestrian_lane         |
| 9  | road_line_solid         |
| 10 | road_line_broken        |
| 11 | road_line_double_yellow |
| 12 | rumble_strip            |

---

## YOLO Label Format
Each `.txt` label file has one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are **normalized** (0.0 to 1.0 relative to image size).

Example `VID_1_000042.txt`:
```
0 0.52 0.48 0.30 0.20
1 0.75 0.40 0.15 0.10
5 0.20 0.60 0.10 0.25
```

---

## Training (on Kaggle)
After annotation, zip and upload `dataset/` to Kaggle:
```python
!pip install ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='/kaggle/input/your-dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project='/kaggle/working/runs'
)
```

---

## Notes
- Raw videos are 720p, frames are resized to 640×640 during extraction
- JPEG quality 85 balances file size and detail
- Seed is fixed (42) for reproducible train/val/test splits
- Multiple classes per image is expected and encouraged
# smart-road-object-detector
