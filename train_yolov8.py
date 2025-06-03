import os
import yaml
import torch
import multiprocessing
from ultralytics import YOLO

# Create dataset configuration
def create_dataset_yaml():
    # Get the absolute path to the current working directory
    workspace_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.join(workspace_dir, 'cricket_dataset')
    
    yaml_content = {
        'path': dataset_dir,  # Absolute path to root directory
        'train': 'images/train',      # Train images
        'val': 'images/val',          # Validation images
        'test': 'images/test',        # Test images (optional)
        
        # Classes
        'names': {
            0: 'Ball',
            1: 'Stumps', 
            2: 'Player',
            3: 'Umpire'
        }
    }
    
    # Create directory for validation set
    os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)
    
    # Write YAML file
    yaml_path = os.path.join(dataset_dir, 'cricket.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print("Created dataset YAML configuration")
    return yaml_path

# Split dataset into train/val sets (80/20 split)
def split_dataset():
    import shutil
    import random
    
    # Get the absolute path to the dataset
    workspace_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.join(workspace_dir, 'cricket_dataset')
    
    # Get all training images
    train_images_dir = os.path.join(dataset_dir, 'images', 'train')
    train_images = os.listdir(train_images_dir)
    train_images = [f for f in train_images if f.endswith('.jpg')]
    
    # Shuffle and split
    random.shuffle(train_images)
    split_idx = int(len(train_images) * 0.8)
    train_set = train_images[:split_idx]
    val_set = train_images[split_idx:]
    
    print(f"Splitting dataset: {len(train_set)} training images, {len(val_set)} validation images")
    
    # Move validation images and their corresponding labels
    for img_file in val_set:
        # Move image
        src_img = os.path.join(train_images_dir, img_file)
        dst_img = os.path.join(dataset_dir, 'images', 'val', img_file)
        shutil.copy(src_img, dst_img)
        
        # Move label if it exists
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(dataset_dir, 'labels', 'train', label_file)
        dst_label = os.path.join(dataset_dir, 'labels', 'val', label_file)
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
    
    print("Dataset split complete")

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
    
    # Create the dataset configuration
    yaml_path = create_dataset_yaml()
    
    # Split the dataset into train/val
    split_dataset()
    
    # Get absolute path to the workspace
    workspace_dir = os.path.abspath(os.getcwd())
    
    # Train the model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Load a pretrained model
    
    # Check initial device
    print(f"Initial model device: {model.device}")
    
    # Explicitly move to CUDA if available
    if gpu_available:
        print("Moving model to CUDA...")
        model = model.to('cuda')
        print(f"Model device after move: {model.device}")
    
    # Train the model
    print(f"Starting training on device: {device}")
    try:
        results = model.train(
            data=yaml_path,
            epochs=100,
            imgsz=640,           # Reduced from 1280 to 640 for memory efficiency
            batch=4,             # Reduced from 8 to 4 for memory efficiency
            patience=20,
            device=device,
            project='cricket_detection',
            name='training_run',
            pretrained=True,
            optimizer='Adam',
            mosaic=1.0,
            scale=0.5,
            fliplr=0.5,
            degrees=15.0,
            translate=0.1,
            perspective=0.0015,
            shear=10.0,
            mixup=0.1,
            cache=False
        )
        print("Training complete!")
        
        # Print the path to the best model
        best_model_path = os.path.join(workspace_dir, 'cricket_detection', 'training_run', 'weights', 'best.pt')
        print(f"Best model saved at: {best_model_path}")
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    # Add freeze_support for Windows
    multiprocessing.freeze_support()
    main() 