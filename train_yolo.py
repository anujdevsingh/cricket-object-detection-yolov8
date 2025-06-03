import os
import torch
from ultralytics import YOLO
import multiprocessing

def main():
    # Check CUDA availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available! Using GPU: {device_name}")
        device = 0  # Use CUDA device 0
    else:
        print("CUDA is not available. Training will use CPU (much slower)")
        device = 'cpu'

    # Print PyTorch version for debugging
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    # Configure paths
    WORKSPACE_DIR = os.path.abspath(os.getcwd())
    DATASET_PATH = os.path.join(WORKSPACE_DIR, "cricket_dataset", "cricket.yaml")

    # Create a model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # load a pretrained model (smaller model)

    # Check initial device
    print(f"Initial model device: {model.device}")

    # Explicitly move to CUDA if available
    if gpu_available:
        print("Moving model to CUDA...")
        model = model.to('cuda')
        print(f"Model device after move: {model.device}")

    # Start training
    print(f"Starting training on device: {device}")
    try:
        # Train the model with reduced memory footprint
        results = model.train(
            data=DATASET_PATH,
            epochs=50,
            imgsz=640,          # Reduced from 1280 to 640
            batch=4,            # Reduced from 16 to 4
            patience=20,
            device=device,
            project='cricket_detection',
            name='training_run_small'
        )
        print("Training completed successfully!")
        print(f"Best model saved at: {os.path.join('cricket_detection', 'training_run_small', 'weights', 'best.pt')}")
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    # Add freeze_support for Windows
    multiprocessing.freeze_support()
    main() 