# Dataset Structure Guide

This document explains how to organize your cricket dataset for training the YOLOv8 model.

## Directory Structure

```
cricket_dataset/
├── cricket.yaml          # Dataset configuration file
├── images/               # All image files
│   ├── train/           # Training images (80% of dataset)
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/             # Validation images (20% of dataset)
│   │   ├── img_101.jpg
│   │   ├── img_102.jpg
│   │   └── ...
│   └── test/            # Test images (optional)
│       ├── img_201.jpg
│       └── ...
└── labels/              # YOLO format annotation files
    ├── train/           # Training labels
    │   ├── img_001.txt
    │   ├── img_002.txt
    │   └── ...
    ├── val/             # Validation labels
    │   ├── img_101.txt
    │   ├── img_102.txt
    │   └── ...
    └── test/            # Test labels (optional)
        ├── img_201.txt
        └── ...
```

## Label Format

Each `.txt` file contains annotations in YOLO format:
```
class_id center_x center_y width height confidence
```

### Class IDs:
- `0`: Ball
- `1`: Stumps
- `2`: Player
- `3`: Umpire

### Coordinate Format:
- All coordinates are normalized (0.0 to 1.0)
- `center_x`, `center_y`: Center point of bounding box
- `width`, `height`: Width and height of bounding box

### Example annotation file (img_001.txt):
```
2 0.5 0.3 0.1 0.4
0 0.8 0.6 0.02 0.02
1 0.2 0.9 0.05 0.2
```

This means:
- Player at center (0.5, 0.3) with size (0.1, 0.4)
- Ball at center (0.8, 0.6) with size (0.02, 0.02)
- Stumps at center (0.2, 0.9) with size (0.05, 0.2)

## Tips for Dataset Creation

1. **Balanced Dataset**: Try to have roughly equal numbers of each class
2. **Diverse Conditions**: Include images with different lighting, angles, and backgrounds
3. **Quality**: Ensure images are clear and objects are clearly visible
4. **Annotations**: Double-check that bounding boxes are accurate and tight around objects
5. **File Naming**: Use consistent naming conventions for images and labels

## Using the Annotation Tool

The included `label_tool.py` will help you create these annotations:

1. Run `python label_tool.py`
2. Select class with keys 0-3
3. Draw bounding boxes by clicking and dragging
4. Save with 's' key
5. Navigate with 'n' (next) and 'p' (previous)

The tool will automatically create the proper directory structure and save annotations in YOLO format. 