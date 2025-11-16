"""
Dataset verification script to check the converted YOLOv11 dataset
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

def verify_dataset(dataset_path="car_damage_yolo_dataset"):
    """Verify the converted dataset structure and content"""
    dataset_path = Path(dataset_path)
    
    print("🔍 Dataset Verification Report")
    print("=" * 50)
    
    # Check directory structure
    required_dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    print("📁 Directory Structure:")
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        exists = "✅" if dir_path.exists() else "❌"
        count = len(list(dir_path.glob("*"))) if dir_path.exists() else 0
        print(f"  {exists} {dir_name}: {count} files")
    
    # Check data.yaml
    yaml_path = dataset_path / "data.yaml"
    yaml_exists = "✅" if yaml_path.exists() else "❌"
    print(f"\n📋 Configuration: {yaml_exists} data.yaml")
    
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            content = f.read()
            if 'nc: 7' in content:
                print("  ✅ 7 classes detected")
            else:
                print("  ❌ Class count mismatch")
    
    # Check label format
    train_labels_dir = dataset_path / "labels/train"
    if train_labels_dir.exists():
        label_files = list(train_labels_dir.glob("*.txt"))
        if label_files:
            sample_label = random.choice(label_files)
            print(f"\n📝 Sample Label File: {sample_label.name}")
            
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                print(f"  📊 Number of annotations: {len(lines)}")
                
                if lines:
                    first_line = lines[0].strip().split()
                    class_id = first_line[0]
                    coords = first_line[1:]
                    print(f"  🏷️  First annotation - Class: {class_id}, Coordinates: {len(coords)} points")
                    
                    # Verify format
                    if len(coords) % 2 == 0 and len(coords) >= 6:
                        print("  ✅ Valid polygon format")
                    else:
                        print("  ❌ Invalid polygon format")
    
    print(f"\n✅ Dataset verification completed!")

def visualize_sample(dataset_path="car_damage_yolo_dataset", num_samples=3):
    """Visualize some sample images with annotations"""
    dataset_path = Path(dataset_path)
    
    # Class names
    classes = ['car-damage', 'crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,128,128)]
    
    train_images_dir = dataset_path / "images/train"
    train_labels_dir = dataset_path / "labels/train"
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        print("❌ Training directories not found")
        return
    
    image_files = list(train_images_dir.glob("*"))[:num_samples]
    
    print(f"\n🖼️  Visualizing {num_samples} sample images...")
    
    for img_file in image_files:
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        # Load corresponding label
        label_file = train_labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            continue
            
        print(f"\n📸 Image: {img_file.name} ({width}x{height})")
        
        # Draw annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class + 3 points
                continue
                
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i+1] * height)
                points.append((x, y))
            
            # Draw polygon
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.polylines(img, [pts], True, colors[class_id], 2)
                
                # Add class label
                cv2.putText(img, classes[class_id], points[0], 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
        
        # Save visualization
        output_file = f"sample_visualization_{img_file.stem}.jpg"
        cv2.imwrite(output_file, img)
        print(f"  💾 Saved visualization: {output_file}")

if __name__ == "__main__":
    # Verify dataset
    verify_dataset()
    
    # Create sample visualizations
    try:
        visualize_sample()
    except ImportError:
        print("\n⚠️  OpenCV not available for visualization")
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")
    
    print("\n🎯 Ready for YOLOv11 training!")
    print("Run: yolo segment train data=car_damage_yolo_dataset/data.yaml model=yolo11n-seg.pt epochs=100")