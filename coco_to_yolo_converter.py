"""
COCO to YOLOv11 Segmentation Format Converter
This script downloads a COCO dataset from Roboflow and converts it to YOLOv11 segmentation format.
"""

import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from roboflow import Roboflow


class COCOToYOLOConverter:
    def __init__(self, output_dir="yolo_dataset"):
        self.output_dir = Path(output_dir)
        self.classes = []
        
    def download_dataset(self):
        """Download the dataset from Roboflow"""
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key="JDkzcoufWk2SbnuraJrf")
        project = rf.workspace("car-damage-type-huesk").project("cardd-bixlm-atwdy")
        version = project.version(1)
        dataset = version.download("coco-segmentation")
        return dataset.location
    
    def setup_yolo_structure(self):
        """Create YOLOv11 directory structure"""
        print("Setting up YOLO directory structure...")
        
        # Create main directories
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val", 
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def polygon_to_normalized_coords(self, polygon, img_width, img_height):
        """Convert polygon coordinates to normalized format for YOLO"""
        normalized_coords = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            normalized_coords.extend([x, y])
        return normalized_coords
    
    def convert_coco_to_yolo(self, coco_dataset_path):
        """Convert COCO annotations to YOLO format"""
        coco_path = Path(coco_dataset_path)
        
        # Process each split (train, valid, test)
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            print(f"Processing {split} split...")
            
            # Map 'valid' to 'val' for YOLO structure
            yolo_split = 'val' if split == 'valid' else split
            
            # Paths for current split
            images_dir = coco_path / split
            annotations_file = coco_path / split / "_annotations.coco.json"
            
            if not annotations_file.exists():
                print(f"Annotations file not found for {split}: {annotations_file}")
                continue
                
            if not images_dir.exists():
                print(f"Images directory not found for {split}: {images_dir}")
                continue
            
            # Load COCO annotations
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            # Extract class information
            if not self.classes:
                self.classes = [cat['name'] for cat in coco_data['categories']]
                print(f"Found classes: {self.classes}")
            
            # Create mapping from category_id to class_index
            cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
            
            # Create mapping from image_id to image info
            img_id_to_info = {img['id']: img for img in coco_data['images']}
            
            # Group annotations by image_id
            img_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_annotations:
                    img_annotations[img_id] = []
                img_annotations[img_id].append(ann)
            
            # Process each image
            for img_id, img_info in img_id_to_info.items():
                img_filename = img_info['file_name']
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Copy image to YOLO structure
                src_img_path = images_dir / img_filename
                dst_img_path = self.output_dir / "images" / yolo_split / img_filename
                
                if src_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)
                else:
                    print(f"Warning: Image not found: {src_img_path}")
                    continue
                
                # Create YOLO annotation file
                txt_filename = Path(img_filename).stem + '.txt'
                txt_path = self.output_dir / "labels" / yolo_split / txt_filename
                
                yolo_annotations = []
                
                # Process annotations for this image
                if img_id in img_annotations:
                    for ann in img_annotations[img_id]:
                        if 'segmentation' not in ann or not ann['segmentation']:
                            continue
                            
                        class_idx = cat_id_to_idx.get(ann['category_id'], 0)
                        
                        # Handle polygon segmentation
                        for segmentation in ann['segmentation']:
                            if len(segmentation) < 6:  # Need at least 3 points (6 coordinates)
                                continue
                                
                            # Convert to normalized coordinates
                            normalized_coords = self.polygon_to_normalized_coords(
                                segmentation, img_width, img_height
                            )
                            
                            # Create YOLO annotation line
                            yolo_line = f"{class_idx} " + " ".join(f"{coord:.6f}" for coord in normalized_coords)
                            yolo_annotations.append(yolo_line)
                
                # Write YOLO annotation file
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    def create_data_yaml(self):
        """Create data.yaml file for YOLOv11"""
        yaml_content = f"""# YOLOv11 dataset configuration
path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(self.classes)}
names: {self.classes}
"""
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created data.yaml at: {yaml_path}")
    
    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "="*50)
        print("CONVERSION SUMMARY")
        print("="*50)
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / "images" / split
            label_dir = self.output_dir / "labels" / split
            
            if img_dir.exists() and label_dir.exists():
                img_count = len(list(img_dir.glob("*")))
                label_count = len(list(label_dir.glob("*.txt")))
                print(f"{split.upper()}: {img_count} images, {label_count} labels")
        
        print(f"\nClasses ({len(self.classes)}): {', '.join(self.classes)}")
        print(f"\nDataset saved to: {self.output_dir.absolute()}")
        print(f"Configuration file: {(self.output_dir / 'data.yaml').absolute()}")
    
    def convert(self):
        """Main conversion function"""
        print("Starting COCO to YOLOv11 conversion...")
        
        # Download dataset
        dataset_path = self.download_dataset()
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Setup YOLO directory structure
        self.setup_yolo_structure()
        
        # Convert annotations
        self.convert_coco_to_yolo(dataset_path)
        
        # Create data.yaml
        self.create_data_yaml()
        
        # Print summary
        self.print_summary()
        
        print("\nConversion completed successfully! 🎉")


if __name__ == "__main__":
    converter = COCOToYOLOConverter()
    converter.convert()