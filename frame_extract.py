import cv2
import os
import glob

# Directory containing videos
video_dir = "side view bowler"
output_dir = "cricket_dataset/images/train"
os.makedirs(output_dir, exist_ok=True)

# Find all MP4 files in the video directory
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
print(f"Found {len(video_files)} video files")

total_saved = 0
frame_interval = 15  # Extract 1 frame every 0.5 seconds (assuming 30fps)

for video_file in video_files:
    video_name = os.path.basename(video_file).split('.')[0]
    print(f"Processing video: {video_name}")
    
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found!")
        continue
        
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'!")
        continue
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {total_frames} frames, {fps} FPS")
    
    frame_id = 0
    video_saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_id % frame_interval == 0:
            filename = os.path.join(output_dir, f"{video_name}_frame_{frame_id:04d}.jpg")
            cv2.imwrite(filename, frame)
            video_saved += 1
            
        frame_id += 1
        
    cap.release()
    total_saved += video_saved
    print(f"Saved {video_saved} frames from {video_name}")
    
print(f"Task completed: Extracted {total_saved} frames total from {len(video_files)} videos")
