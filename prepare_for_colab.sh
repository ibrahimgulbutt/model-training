#!/bin/bash

# 📦 Google Drive Upload Preparation Script
# This script helps you prepare your dataset for Google Colab training

echo "🚀 Preparing dataset for Google Colab..."
echo "========================================="

# Check if dataset exists
if [ ! -d "car_damage_yolo_dataset" ]; then
    echo "❌ Error: car_damage_yolo_dataset folder not found!"
    echo "Please run the conversion script first:"
    echo "python convert_dataset.py"
    exit 1
fi

echo "✅ Dataset found"

# Create upload package
UPLOAD_DIR="YOLOv11_Car_Damage"
echo "📁 Creating upload package: $UPLOAD_DIR"

# Create directory structure
mkdir -p $UPLOAD_DIR/models

# Copy dataset
echo "📋 Copying dataset..."
cp -r car_damage_yolo_dataset $UPLOAD_DIR/

# Copy training notebook
echo "📓 Copying training notebook..."
cp YOLOv11_Car_Damage_Training.ipynb $UPLOAD_DIR/

# Copy documentation
echo "📚 Copying documentation..."
cp Google_Colab_Training_Guide.md $UPLOAD_DIR/
cp README.md $UPLOAD_DIR/

# Create a setup instructions file
cat > $UPLOAD_DIR/UPLOAD_INSTRUCTIONS.txt << EOF
🚀 Google Colab Setup Instructions
==================================

1. UPLOAD TO GOOGLE DRIVE:
   - Upload this entire 'YOLOv11_Car_Damage' folder to your Google Drive
   - Place it in the root directory: My Drive/YOLOv11_Car_Damage/

2. OPEN COLAB NOTEBOOK:
   - Go to https://colab.research.google.com/
   - Upload the YOLOv11_Car_Damage_Training.ipynb file
   - Or open it directly from Google Drive

3. ENABLE GPU:
   - In Colab: Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (free tier)

4. RUN THE NOTEBOOK:
   - Run cells one by one
   - Start with the test training (1 epoch)
   - Then run full training if test works

5. MONITOR TRAINING:
   - Check GPU usage to avoid disconnections
   - Keep browser tab active
   - Training takes 2-4 hours

6. DOWNLOAD RESULTS:
   - Best model: best_model.pt
   - Training curves: training_curves.png
   - Complete package: car_damage_results.zip

Dataset Info:
- Training Images: 9,600
- Validation Images: 800
- Classes: 7 (car damage types)
- Expected mAP50: 0.6-0.8

Troubleshooting:
- If GPU memory error: reduce batch size to 8 or 4
- If disconnection: save frequently to Google Drive
- If slow training: use image caching (cache=True)

Good luck with your training! 🎯
EOF

# Calculate sizes
DATASET_SIZE=$(du -sh car_damage_yolo_dataset | cut -f1)
UPLOAD_SIZE=$(du -sh $UPLOAD_DIR | cut -f1)

echo ""
echo "📊 Package Summary:"
echo "==================="
echo "📁 Dataset size: $DATASET_SIZE"
echo "📦 Upload package size: $UPLOAD_SIZE"
echo ""
echo "📋 Contents:"
echo "├── car_damage_yolo_dataset/     # Your converted dataset"
echo "├── YOLOv11_Car_Damage_Training.ipynb  # Colab notebook"
echo "├── Google_Colab_Training_Guide.md     # Detailed guide"
echo "├── README.md                          # Project info"
echo "├── models/                            # (empty, for results)"
echo "└── UPLOAD_INSTRUCTIONS.txt           # Setup instructions"

echo ""
echo "✅ Upload package ready: $UPLOAD_DIR/"
echo ""
echo "🔄 Next Steps:"
echo "1. Upload the '$UPLOAD_DIR' folder to Google Drive"
echo "2. Open YOLOv11_Car_Damage_Training.ipynb in Google Colab"
echo "3. Follow the instructions in the notebook"
echo ""
echo "💡 Tip: You can also create a ZIP file for easier upload:"
echo "   zip -r YOLOv11_Car_Damage.zip $UPLOAD_DIR/"
echo ""

# Offer to create ZIP
read -p "📦 Create ZIP file for upload? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗜️ Creating ZIP file..."
    zip -r YOLOv11_Car_Damage.zip $UPLOAD_DIR/ > /dev/null 2>&1
    ZIP_SIZE=$(du -sh YOLOv11_Car_Damage.zip | cut -f1)
    echo "✅ ZIP created: YOLOv11_Car_Damage.zip ($ZIP_SIZE)"
    echo "📤 You can now upload this ZIP to Google Drive and extract it there"
fi

echo ""
echo "🎉 Setup complete! Happy training on Google Colab! 🚀"