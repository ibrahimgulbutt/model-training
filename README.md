# COCO to YOLOv11 Segmentation Dataset Converter

This project converts a COCO segmentation dataset from Roboflow to YOLOv11 format for car damage detection.

## Dataset Information

- **Source**: Roboflow car damage detection dataset
- **Original Format**: COCO segmentation
- **Target Format**: YOLOv11 segmentation
- **Classes**: 7 types of car damage
  1. car-damage
  2. crack
  3. dent
  4. glass shatter
  5. lamp broken
  6. scratch
  7. tire flat

## Dataset Statistics

- **Training Images**: 9,600
- **Validation Images**: 800
- **Total Images**: 10,400

## Directory Structure

```
car_damage_yolo_dataset/
├── data.yaml                 # YOLOv11 configuration file
├── images/
│   ├── train/               # Training images
│   ├── val/                 # Validation images
│   └── test/                # Test images (empty in this dataset)
└── labels/
    ├── train/               # Training labels (polygon coordinates)
    ├── val/                 # Validation labels (polygon coordinates)
    └── test/                # Test labels (empty)
```

## Files in This Project

1. **`coco_to_yolo_converter.py`** - Main conversion class with all the conversion logic
2. **`convert_dataset.py`** - Simple script to run the conversion
3. **`car_damage_yolo_dataset/`** - The converted dataset ready for YOLOv11 training

## Usage

### Running the Conversion (Already Done)
```bash
python convert_dataset.py
```

### Training with YOLOv11

1. **Install YOLOv11**:
```bash
pip install ultralytics
```

2. **Train the model**:
```bash
# Basic training
yolo segment train data=car_damage_yolo_dataset/data.yaml model=yolo11n-seg.pt epochs=100

# Advanced training with custom parameters
yolo segment train data=car_damage_yolo_dataset/data.yaml model=yolo11s-seg.pt epochs=200 imgsz=640 batch=16
```

3. **Model sizes available**:
   - `yolo11n-seg.pt` - Nano (fastest, smallest)
   - `yolo11s-seg.pt` - Small
   - `yolo11m-seg.pt` - Medium  
   - `yolo11l-seg.pt` - Large
   - `yolo11x-seg.pt` - Extra Large (most accurate)

### Validation and Testing

```bash
# Validate the trained model
yolo segment val model=runs/segment/train/weights/best.pt data=car_damage_yolo_dataset/data.yaml

# Run inference on new images
yolo segment predict model=runs/segment/train/weights/best.pt source=path/to/images/
```

## Label Format

The YOLOv11 segmentation format uses normalized polygon coordinates:
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: Integer index of the class (0-6 for the 7 damage types)
- `x1 y1 x2 y2 ...`: Normalized polygon coordinates (0.0 to 1.0)

## Configuration File (data.yaml)

The `data.yaml` file contains:
- Dataset path
- Train/val/test splits
- Number of classes (nc: 7)
- Class names list

## Tips for Training

1. **Start with a pretrained model** for better results
2. **Adjust batch size** based on your GPU memory
3. **Use data augmentation** (automatically applied by YOLOv11)
4. **Monitor training** with TensorBoard logs in `runs/segment/train/`
5. **Validate regularly** to check for overfitting

## Requirements

- Python 3.7+
- roboflow
- ultralytics (for training)
- pycocotools
- opencv-python
- numpy
- Pillow

## Author

Converted using the COCO to YOLOv11 converter script.