import cv2
import time
import numpy as np
import os
import sys
from ultralytics import YOLO

# Check if supervision is installed, if not provide instructions
try:
    import supervision as sv
except ImportError:
    print("Error: The 'supervision' package is required but not installed.")
    print("Please install it with: pip install supervision")
    print("For more info: https://github.com/roboflow/supervision")
    sys.exit(1)

def process_video(model_path, video_path, output_path=None, conf_threshold=0.25):
    """
    Process a video with YOLOv8 model and create annotated output
    
    Args:
        model_path: Path to the trained YOLOv8 model
        video_path: Path to the input video
        output_path: Path for the output video (default: based on input name)
        conf_threshold: Confidence threshold for detections
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Class names
    class_names = ['Ball', 'Stumps', 'Player', 'Umpire']
    
    # Create annotator with default parameters
    box_annotator = sv.BoxAnnotator(thickness=2)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output path if not specified
    if output_path is None:
        video_name = os.path.basename(video_path)
        name, ext = os.path.splitext(video_name)
        output_path = f"{name}_annotated{ext}"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create object tracker
    tracker = sv.ByteTrack()
    
    # For statistics
    class_counts = {name: 0 for name in class_names}
    frame_counts = {name: [] for name in class_names}
    
    # Process the video
    frame_idx = 0
    start_time = time.time()
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 inference
        results = model(frame, conf=conf_threshold)[0]
        
        # Get detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Track objects
        if len(detections) > 0:
            detections = tracker.update_with_detections(detections)
        
        # Update statistics
        for i, confidence in enumerate(detections.confidence):
            class_id = detections.class_id[i]
            class_name = class_names[class_id]
            class_counts[class_name] += 1
        
        # Count objects per class in this frame
        for class_name in class_names:
            count = sum(1 for i, class_id in enumerate(detections.class_id) 
                        if class_names[class_id] == class_name)
            frame_counts[class_name].append(count)
        
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Draw bounding boxes manually
        if len(detections) > 0:
            for i, (xyxy, confidence, class_id) in enumerate(zip(
                detections.xyxy, detections.confidence, detections.class_id
            )):
                # Extract coordinates
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get class name
                class_name = class_names[class_id]
                
                # Draw rectangle
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 0), 
                    2
                )
                
                # Create label text
                label = f"{class_name} {confidence:.2f}"
                
                # Add tracker ID if available
                if detections.tracker_id is not None:
                    label += f" ID:{detections.tracker_id[i]}"
                
                # Get text size for background rectangle
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )[0]
                
                # Draw text background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        # Add frame information
        frame_info = f"Frame: {frame_idx}/{frame_count} | "
        frame_info += " | ".join([f"{name}: {frame_counts[name][-1]}" for name in class_names])
        cv2.putText(annotated_frame, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write the frame
        out.write(annotated_frame)
        
        # Update progress every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (frame_count - frame_idx - 1) / fps_processing if fps_processing > 0 else 0
            print(f"Progress: {frame_idx+1}/{frame_count} frames "
                  f"({(frame_idx+1)/frame_count*100:.1f}%) | "
                  f"Processing speed: {fps_processing:.2f} FPS | "
                  f"ETA: {remaining:.1f} seconds")
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Calculate statistics
    avg_counts = {name: np.mean(frame_counts[name]) for name in class_names}
    max_counts = {name: np.max(frame_counts[name]) for name in class_names}
    
    # Print statistics
    print("\nProcessing complete!")
    print(f"Total frames processed: {frame_idx}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print("\nDetection Statistics:")
    for name in class_names:
        print(f"  {name}: Total: {class_counts[name]}, Avg per frame: {avg_counts[name]:.2f}, Max: {max_counts[name]}")
    
    print(f"\nAnnotated video saved to: {output_path}")
    
    return output_path

def main():
    # Check if model exists (use absolute path for consistency)
    workspace_dir = os.path.abspath(os.getcwd())
    model_path = os.path.join(workspace_dir, "cricket_detection", "training_run7", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure training has completed successfully.")
        
        # Check for alternative locations
        alt_path = os.path.join(workspace_dir, "runs", "detect", "train", "weights", "best.pt")
        if os.path.exists(alt_path):
            print(f"Found alternative model at: {alt_path}")
            model_path = alt_path
        else:
            return
    
    # Directory containing videos
    video_dir = "front view"
    
    # Create output directory
    output_dir = "annotated_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MP4 files in the video directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in '{video_dir}'")
        return
    
    print(f"Found {len(video_files)} videos to process")
    print(f"Using model: {model_path}")
    
    # Process first video only for testing (comment out in production)
    # video_file = video_files[0]
    # print(f"\nProcessing video: {video_file}")
    # video_path = os.path.join(video_dir, video_file)
    # output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_annotated.mp4")
    # process_video(model_path, video_path, output_path)
    
    
    # Uncomment below to process all videos
    
    # Process each video
    for i, video_file in enumerate(video_files):
        print(f"\nProcessing video {i+1}/{len(video_files)}: {video_file}")
        
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_annotated.mp4")
        
        process_video(model_path, video_path, output_path)
    
    print("\nAll videos processed!")

if __name__ == "__main__":
    main() 