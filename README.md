# Cricket Object Detection with YOLOv8

A comprehensive computer vision project for detecting and tracking cricket-related objects in videos using YOLOv8. This system can detect and track:

- **Ball** - Cricket ball detection and tracking
- **Stumps** - Wicket stumps identification  
- **Players** - Cricket players on the field
- **Umpires** - Match officials detection

## 🚀 Features

- **Multi-Object Detection**: Simultaneous detection of 4 different cricket objects
- **Real-time Tracking**: Uses ByteTrack for consistent object tracking across frames
- **Video Processing**: Full video annotation with bounding boxes and labels
- **Statistics**: Detailed object count statistics and frame-by-frame analysis
- **Easy Annotation**: Interactive tool for labeling training data
- **Flexible Training**: Custom YOLOv8 training pipeline with validation

## 📁 Project Structure

```
├── frame_extract.py        # Extract frames from video files
├── label_tool.py           # Interactive annotation tool for labeling frames
├── train_yolov8.py         # Train YOLOv8 model on annotated dataset
├── annotate_video.py       # Process videos with trained model
├── requirements.txt        # Python dependencies
├── cricket_dataset/        # Dataset directory
│   ├── images/            # Training and validation images
│   │   ├── train/         # Training images
│   │   └── val/           # Validation images
│   ├── labels/            # YOLO format annotations
│   │   ├── train/         # Training labels
│   │   └── val/           # Validation labels
│   └── cricket.yaml       # Dataset configuration
└── annotated_videos/       # Output directory for processed videos
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cricket-object-detection-yolov8.git
cd cricket-object-detection-yolov8
```

### 2. Create Virtual Environment
```bash
python -m venv cricketenv
# On Windows:
cricketenv\Scripts\activate
# On macOS/Linux:
source cricketenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model (Optional)
The system will automatically download YOLOv8 base models when needed, but you can pre-download:
```bash
# This will be done automatically on first run
```

## 📖 Usage Guide

### Step 1: Extract Frames from Videos
Extract frames from your cricket videos for annotation:

```bash
python frame_extract.py
```

- Place your video files in the project directory
- The script will extract frames at regular intervals
- Frames are saved to `cricket_dataset/images/` directory

### Step 2: Annotate the Extracted Frames
Use the interactive annotation tool to label objects:

```bash
python label_tool.py
```

**Annotation Controls:**
- **'n'**: Next image
- **'p'**: Previous image  
- **'0'**: Select Ball class
- **'1'**: Select Stumps class
- **'2'**: Select Player class
- **'3'**: Select Umpire class
- **'s'**: Save current annotations
- **'c'**: Clear all annotations for current image
- **'q'**: Quit annotation tool

**How to Annotate:**
1. Select a class (0-3)
2. Click and drag to draw bounding boxes around objects
3. Save regularly with 's'
4. Navigate between images with 'n' and 'p'

### Step 3: Train the YOLOv8 Model
Train your custom object detection model:

```bash
python train_yolov8.py
```

**Training Features:**
- Automatic train/validation split
- Progress monitoring with metrics
- Model checkpoints saved automatically
- Best model saved to `cricket_detection/training_run/weights/best.pt`

### Step 4: Process Videos with Trained Model
Annotate new videos using your trained model:

```bash
python annotate_video.py
```

**Output Features:**
- Bounding boxes around detected objects
- Class labels and confidence scores
- Object tracking IDs
- Frame-by-frame statistics
- Progress monitoring during processing

## 🎯 Model Performance

The system is designed to detect:
- **Ball**: Small, fast-moving cricket ball
- **Stumps**: Wooden wicket posts
- **Players**: Cricket players in various poses
- **Umpires**: Match officials on the field

## 🔧 Configuration

### Dataset Configuration
Edit `cricket_dataset/cricket.yaml` to modify:
- Dataset paths
- Class names
- Training parameters

### Model Parameters
Modify training parameters in `train_yolov8.py`:
- Epochs, batch size, learning rate
- Image size, confidence threshold
- Validation split ratio

## 📊 Output

### Annotated Videos
- Saved to `annotated_videos/` directory
- Include bounding boxes, labels, and tracking IDs
- Real-time statistics overlay

### Training Metrics
- Model performance metrics
- Validation accuracy
- Loss curves and mAP scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 👨‍💻 Author

**Anuj Dev Singh**
- Project Creator & Lead Developer


## 📋 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Supervision (for tracking)
- NumPy
- PyYAML

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size in training
2. **Video Not Opening**: Check video codec compatibility
3. **Model Not Found**: Ensure training completed successfully
4. **Annotation Tool Issues**: Check OpenCV installation

### Getting Help:
- Open an issue on GitHub
- Check existing issues for solutions
- Ensure all dependencies are correctly installed

## 🚀 Future Improvements

- [ ] Real-time camera feed processing
- [ ] Advanced tracking algorithms
- [ ] Mobile app integration
- [ ] Web-based annotation interface
- [ ] Automated player action recognition
- [ ] Match statistics generation

---

**Note**: This project requires significant computing resources for training. For best results, use a CUDA-compatible GPU. 