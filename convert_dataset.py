#!/usr/bin/env python3
"""
Simple script to convert your COCO dataset to YOLOv11 format
"""

from coco_to_yolo_converter import COCOToYOLOConverter

def main():
    print("🚗 Car Damage Detection Dataset Converter")
    print("Converting COCO to YOLOv11 segmentation format...")
    print("-" * 50)
    
    # Initialize converter
    converter = COCOToYOLOConverter(output_dir="car_damage_yolo_dataset")
    
    # Run conversion
    converter.convert()
    
    print("\n✅ Your dataset is ready for YOLOv11 training!")
    print("\nTo train with YOLOv11:")
    print("1. Install ultralytics: pip install ultralytics")
    print("2. Train: yolo segment train data=car_damage_yolo_dataset/data.yaml model=yolo11n-seg.pt epochs=100")

if __name__ == "__main__":
    main()