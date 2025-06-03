import cv2
import os
import numpy as np
import glob

# Define the classes we want to detect
class_names = ['Ball', 'Stumps', 'Player', 'Umpire']

# Constants
IMAGES_DIR = 'cricket_dataset/images/train'
LABELS_DIR = 'cricket_dataset/labels/train'
os.makedirs(LABELS_DIR, exist_ok=True)

# Colors for visualization (in BGR format)
COLORS = {
    'Ball': (0, 0, 255),       # Red
    'Stumps': (0, 255, 0),     # Green
    'Player': (255, 0, 0),     # Blue
    'Umpire': (255, 255, 0)    # Cyan
}

# Global variables
current_class = 0
drawing = False
ix, iy = -1, -1
boxes = []
img = None
window_name = 'Cricket Annotation Tool'
current_image_path = ''
scale_factor = 1.0  # For image resizing if needed

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img, boxes, current_class, scale_factor
    
    x = int(x / scale_factor)  # Convert back to original coordinates
    y = int(y / scale_factor)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), COLORS[class_names[current_class]], 2)
            cv2.putText(img_copy, class_names[current_class], (ix, iy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_names[current_class]], 2)
            
            # Resize for display if needed
            display_img = cv2.resize(img_copy, (0, 0), fx=scale_factor, fy=scale_factor) if scale_factor != 1.0 else img_copy
            cv2.imshow(window_name, display_img)
            
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            # Calculate width and height of bounding box
            w = abs(x - ix)
            h = abs(y - iy)
            # Get the top-left corner
            x_min = min(ix, x)
            y_min = min(iy, y)
            
            # Draw the rectangle
            cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), COLORS[class_names[current_class]], 2)
            cv2.putText(img, class_names[current_class], (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_names[current_class]], 2)
            
            # Add the box to our list
            # Format: [class_id, x_center, y_center, width, height] (normalized)
            image_h, image_w = img.shape[:2]
            x_center = (x_min + w/2) / image_w
            y_center = (y_min + h/2) / image_h
            width = w / image_w
            height = h / image_h
            
            boxes.append([current_class, x_center, y_center, width, height])
            
            # Display the updated image
            display_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor) if scale_factor != 1.0 else img
            cv2.imshow(window_name, display_img)

def save_annotations():
    global boxes, current_image_path
    
    if not boxes:
        print("No annotations to save")
        return
        
    # Create YOLO format annotation file
    filename = os.path.splitext(os.path.basename(current_image_path))[0]
    label_path = os.path.join(LABELS_DIR, filename + '.txt')
    
    with open(label_path, 'w') as f:
        for box in boxes:
            # YOLO format: class_id x_center y_center width height
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
            
    print(f"Saved {len(boxes)} annotations to {label_path}")

def main():
    global img, current_class, boxes, current_image_path, scale_factor
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.jpg')))
    
    if not image_files:
        print(f"No images found in {IMAGES_DIR}")
        return
        
    print(f"Found {len(image_files)} images to annotate")
    print("Controls:")
    print("  'n': Next image")
    print("  'p': Previous image")
    print("  '0-3': Select class (0=Ball, 1=Stumps, 2=Player, 3=Umpire)")
    print("  's': Save annotations")
    print("  'c': Clear all annotations for current image")
    print("  'q': Quit")
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    index = 0
    while index < len(image_files):
        current_image_path = image_files[index]
        img = cv2.imread(current_image_path)
        boxes = []
        
        # Check if annotation file already exists
        filename = os.path.splitext(os.path.basename(current_image_path))[0]
        label_path = os.path.join(LABELS_DIR, filename + '.txt')
        
        if os.path.exists(label_path):
            # Load existing annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            image_h, image_w = img.shape[:2]
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                x_min = int((x_center - width/2) * image_w)
                y_min = int((y_center - height/2) * image_h)
                w = int(width * image_w)
                h = int(height * image_h)
                
                # Draw the rectangle
                cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), 
                             COLORS[class_names[class_id]], 2)
                cv2.putText(img, class_names[class_id], (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_names[class_id]], 2)
                
                # Add to boxes list
                boxes.append([class_id, x_center, y_center, width, height])
        
        # Determine if we need to resize for display
        screen_h, screen_w = 900, 1600  # Approximate screen size
        image_h, image_w = img.shape[:2]
        
        if image_w > screen_w or image_h > screen_h:
            scale_w = screen_w / image_w
            scale_h = screen_h / image_h
            scale_factor = min(scale_w, scale_h)
            display_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            scale_factor = 1.0
            display_img = img
        
        cv2.imshow(window_name, display_img)
        print(f"Image {index+1}/{len(image_files)}: {os.path.basename(current_image_path)}")
        print(f"Current class: {class_names[current_class]}")
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('n'):  # Next image
            save_annotations()
            index += 1
        elif key == ord('p') and index > 0:  # Previous image
            save_annotations()
            index -= 1
        elif key == ord('s'):  # Save annotations
            save_annotations()
        elif key == ord('c'):  # Clear annotations
            boxes = []
            img = cv2.imread(current_image_path)
            display_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor) if scale_factor != 1.0 else img
            cv2.imshow(window_name, display_img)
        elif key >= ord('0') and key <= ord('3'):  # Class selection
            current_class = key - ord('0')
            print(f"Selected class: {class_names[current_class]}")
    
    cv2.destroyAllWindows()
    print("Annotation complete!")

if __name__ == "__main__":
    main() 