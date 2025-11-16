# 🚀 YOLOv11 Car Damage Segmentation Training on Google Colab

This guide will help you train your converted car damage detection dataset on Google Colab using free GPU resources.

## 📋 Prerequisites

1. **Google Account** - For accessing Google Colab
2. **Google Drive** - To store your dataset and model weights
3. **Converted Dataset** - Your `car_damage_yolo_dataset` folder

## 🗂️ Step 1: Upload Dataset to Google Drive

### Option A: Direct Upload (Recommended for smaller datasets)

1. **Create a folder structure in Google Drive:**
   ```
   My Drive/
   └── YOLOv11_Car_Damage/
       ├── car_damage_yolo_dataset/
       │   ├── data.yaml
       │   ├── images/
       │   │   ├── train/
       │   │   └── val/
       │   └── labels/
       │       ├── train/
       │       └── val/
       └── models/  (will store trained weights)
   ```

2. **Upload your dataset:**
   - Zip your `car_damage_yolo_dataset` folder
   - Upload to `My Drive/YOLOv11_Car_Damage/`
   - Extract it in the folder

### Option B: Using Google Drive Desktop App

1. Install Google Drive Desktop app
2. Copy your `car_damage_yolo_dataset` folder to the synced Drive folder
3. Wait for synchronization to complete

## 🚀 Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Enable GPU:**
   - Go to `Runtime` → `Change runtime type`
   - Select `Hardware accelerator: GPU`
   - Choose `GPU type: T4` (free tier)
   - Click `Save`

## 📝 Step 3: Colab Training Script

Copy and paste the following code cells into your Colab notebook:

### Cell 1: Setup and Installation

```python
# 🔧 Install required packages
!pip install ultralytics roboflow

# 📊 Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# 📁 Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 📂 Change to working directory
import os
os.chdir('/content/drive/My Drive/YOLOv11_Car_Damage')
print("Current directory:", os.getcwd())
```

### Cell 2: Verify Dataset

```python
# 🔍 Verify dataset structure
import os
from pathlib import Path

dataset_path = Path('car_damage_yolo_dataset')

print("📁 Dataset Structure Verification:")
print("=" * 40)

# Check main directories
dirs_to_check = [
    'images/train', 'images/val', 'labels/train', 'labels/val'
]

for dir_name in dirs_to_check:
    dir_path = dataset_path / dir_name
    if dir_path.exists():
        file_count = len(list(dir_path.glob('*')))
        print(f"✅ {dir_name}: {file_count} files")
    else:
        print(f"❌ {dir_name}: Not found")

# Check data.yaml
yaml_path = dataset_path / 'data.yaml'
if yaml_path.exists():
    print(f"✅ data.yaml: Found")
    with open(yaml_path, 'r') as f:
        print("📄 Configuration:")
        print(f.read())
else:
    print(f"❌ data.yaml: Not found")
```

### Cell 3: Update data.yaml for Colab

```python
# 🔧 Update data.yaml paths for Colab environment
import yaml

# Read current data.yaml
with open('car_damage_yolo_dataset/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Update paths for Colab
data_config['path'] = '/content/drive/My Drive/YOLOv11_Car_Damage/car_damage_yolo_dataset'

# Write updated config
with open('car_damage_yolo_dataset/data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("✅ Updated data.yaml for Colab environment")
print("Updated configuration:")
with open('car_damage_yolo_dataset/data.yaml', 'r') as f:
    print(f.read())
```

### Cell 4: Start Training (Basic)

```python
# 🚀 Basic Training - Start with 1 epoch for testing
from ultralytics import YOLO

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("🏁 Starting YOLOv11 Training...")
print("=" * 50)

# Initialize model
model = YOLO('yolo11n-seg.pt')  # Start with nano model

# Train the model (start with 1 epoch for testing)
results = model.train(
    data='car_damage_yolo_dataset/data.yaml',
    epochs=1,  # Start small to test
    imgsz=640,
    batch=8,   # Adjust based on GPU memory
    name='car_damage_test',
    project='models',
    save=True,
    device=0   # Use GPU
)

print("✅ Test training completed!")
```

### Cell 5: Full Training

```python
# 🏋️ Full Training Session
print("🚀 Starting Full Training...")
print("=" * 50)

# Initialize fresh model
model = YOLO('yolo11n-seg.pt')

# Full training
results = model.train(
    data='car_damage_yolo_dataset/data.yaml',
    epochs=100,        # Full training epochs
    imgsz=640,         # Image size
    batch=16,          # Batch size (adjust if GPU memory issues)
    name='car_damage_full',
    project='models',
    save=True,
    patience=20,       # Early stopping patience
    device=0,          # Use GPU
    workers=2,         # Number of workers
    optimizer='AdamW', # Optimizer
    lr0=0.01,         # Initial learning rate
    cos_lr=True,      # Cosine learning rate scheduler
    augment=True,     # Data augmentation
    cache=True,       # Cache images for faster training
    verbose=True
)

print("🎉 Training completed!")
```

### Cell 6: Training Monitoring

```python
# 📊 Monitor Training Progress
import matplotlib.pyplot as plt
from IPython.display import Image, display
import glob

# Display training curves
results_dir = 'models/car_damage_full'

# Check if results exist
if os.path.exists(results_dir):
    print("📈 Training Results:")
    
    # Display results image if available
    results_img = f"{results_dir}/results.png"
    if os.path.exists(results_img):
        display(Image(results_img))
    
    # Show validation images
    val_images = glob.glob(f"{results_dir}/val_*.jpg")
    if val_images:
        print("🔍 Validation Results:")
        for img_path in val_images[:3]:  # Show first 3
            print(f"📸 {os.path.basename(img_path)}")
            display(Image(img_path))
else:
    print("⚠️ Training results not found. Make sure training completed successfully.")
```

### Cell 7: Model Validation

```python
# ✅ Validate the trained model
print("🔬 Validating Trained Model...")

# Load best model
best_model_path = 'models/car_damage_full/weights/best.pt'

if os.path.exists(best_model_path):
    model = YOLO(best_model_path)
    
    # Validate
    validation_results = model.val(
        data='car_damage_yolo_dataset/data.yaml',
        save_json=True,
        save_hybrid=True
    )
    
    print("📊 Validation Metrics:")
    print(f"mAP50: {validation_results.box.map50:.4f}")
    print(f"mAP50-95: {validation_results.box.map:.4f}")
else:
    print("❌ Best model not found. Check if training completed successfully.")
```

### Cell 8: Test Inference

```python
# 🧪 Test Inference on Sample Images
import random

# Get some test images
test_images = list(Path('car_damage_yolo_dataset/images/val').glob('*'))[:5]

if test_images and os.path.exists(best_model_path):
    model = YOLO(best_model_path)
    
    print("🖼️ Running Inference on Sample Images:")
    
    for img_path in test_images:
        print(f"Processing: {img_path.name}")
        
        # Run inference
        results = model(str(img_path))
        
        # Save results
        for i, result in enumerate(results):
            result.save(filename=f'inference_result_{img_path.stem}.jpg')
            
    print("✅ Inference completed! Check saved images.")
else:
    print("⚠️ No test images found or model not available.")
```

### Cell 9: Download Results

```python
# 💾 Prepare Results for Download
import shutil
from google.colab import files

print("📦 Preparing results for download...")

# Create a results package
results_package = 'car_damage_training_results'
os.makedirs(results_package, exist_ok=True)

# Copy important files
if os.path.exists('models/car_damage_full'):
    # Copy best weights
    if os.path.exists('models/car_damage_full/weights/best.pt'):
        shutil.copy('models/car_damage_full/weights/best.pt', 
                   f'{results_package}/best_model.pt')
    
    # Copy training results
    if os.path.exists('models/car_damage_full/results.png'):
        shutil.copy('models/car_damage_full/results.png', 
                   f'{results_package}/training_results.png')
    
    # Copy validation images
    val_images = glob.glob('models/car_damage_full/val_*.jpg')
    for img in val_images:
        shutil.copy(img, results_package)

# Create a summary file
with open(f'{results_package}/training_summary.txt', 'w') as f:
    f.write("🚗 Car Damage Detection - YOLOv11 Training Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training completed: {os.getcwd()}\n")
    f.write(f"Model: YOLOv11n-seg\n")
    f.write(f"Dataset: Car Damage (7 classes)\n")
    f.write(f"Classes: car-damage, crack, dent, glass shatter, lamp broken, scratch, tire flat\n")
    f.write(f"Training images: 9600\n")
    f.write(f"Validation images: 800\n")

# Zip the results
shutil.make_archive('car_damage_results', 'zip', results_package)

print("✅ Results package created!")
print("📁 Files ready for download:")
print("- car_damage_results.zip (complete package)")
print("- models/car_damage_full/weights/best.pt (best model)")

# Download the zip file
files.download('car_damage_results.zip')
```

## ⚙️ Step 4: Advanced Training Options

### For Better Results (if you have Colab Pro):

```python
# 🔥 Advanced Training with Larger Model
model = YOLO('yolo11s-seg.pt')  # Use small model instead of nano

results = model.train(
    data='car_damage_yolo_dataset/data.yaml',
    epochs=200,
    imgsz=640,
    batch=32,          # Larger batch size
    name='car_damage_advanced',
    project='models',
    mosaic=0.5,        # Mosaic augmentation
    mixup=0.1,         # Mixup augmentation
    copy_paste=0.1,    # Copy-paste augmentation
    degrees=10,        # Rotation augmentation
    translate=0.1,     # Translation augmentation
    scale=0.5,         # Scale augmentation
    fliplr=0.5,        # Horizontal flip
    device=0
)
```

## 🔧 Troubleshooting

### GPU Memory Issues:
```python
# Reduce batch size if you get GPU memory errors
batch_size = 4  # Start with smaller batch
```

### Slow Training:
```python
# Use image caching for faster training
cache = 'ram'  # or 'disk'
```

### Connection Timeout:
```python
# Keep session alive (run this in a separate cell)
import time
while True:
    time.sleep(300)  # Keep alive every 5 minutes
    print("🔄 Keeping session alive...")
```

## 📊 Expected Results

- **Training Time**: 2-4 hours on free Colab GPU
- **mAP50**: Expected 0.6-0.8 (depending on data quality)
- **Model Size**: ~6MB (YOLOv11n-seg)

## 💡 Tips for Better Training

1. **Use Colab Pro** for longer training sessions and better GPUs
2. **Monitor GPU usage** to avoid disconnections
3. **Save checkpoints** regularly
4. **Use smaller image sizes** (416 instead of 640) for faster training
5. **Experiment with data augmentation** parameters

## 📁 File Structure After Training

```
My Drive/YOLOv11_Car_Damage/
├── car_damage_yolo_dataset/
├── models/
│   └── car_damage_full/
│       ├── weights/
│       │   ├── best.pt      # Best model weights
│       │   └── last.pt      # Last checkpoint
│       ├── results.png      # Training curves
│       └── val_*.jpg        # Validation images
└── car_damage_results.zip   # Download package
```

## 🎯 Next Steps After Training

1. **Download your trained model** (`best.pt`)
2. **Test on new images** using the inference script
3. **Deploy the model** for real-world applications
4. **Fine-tune** with more data if needed

## ⚠️ Important Notes

- **Free Colab has usage limits** - you might get disconnected after 12 hours
- **Save your work frequently** to Google Drive
- **Monitor training progress** to catch issues early
- **Keep browser tab active** to maintain connection

---

Happy Training! 🚀 Your car damage detection model will be ready soon!