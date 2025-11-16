# Run YOLOv11 Segmentation Training on Google Colab

This guide walks you through running your YOLOv11 segmentation training on Google Colab using your dataset at:

`/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset`

It includes ready-to-run Colab cells, tips for handling spaces in paths, and troubleshooting.

---

## 1) Prepare Google Colab

1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook (Runtime → New notebook).
3. Set Runtime → Change runtime type → Hardware accelerator: GPU (TPU not supported for this workflow).

---

## 2) Notebook cells (copy & paste sequentially)

Below are the exact cells to paste into Colab. Run them in order.

### Cell 1 — Install dependencies (BASH CELL - use `!` commands)

```bash
# Install ultralytics and helpers
!pip install -U pip
!pip install ultralytics pycocotools opencv-python-headless Pillow pyyaml
```

**IMPORTANT**: This should be a **Code cell** in Colab, and the commands start with `!` (exclamation mark) to run shell commands.

Notes:
- `ultralytics` provides both the `yolo` CLI and Python API (YOLO class). If `yolo` CLI still fails, use the Python API (shown below).
- `opencv-python-headless` is recommended on Colab to avoid GUI dependencies.

---

### Cell 2 — Mount Google Drive (PYTHON CELL)

```python
from google.colab import drive
drive.mount('/content/drive')
```

When prompted, follow the OAuth link and paste the auth code.

---

### Cell 3 — Set working directory (PYTHON CELL)

Your dataset path contains spaces. Use a path variable and `os.chdir()` with the full quoted path.

```python
import os
# Exact path you provided (note spaces)
DRIVE_DATASET_PATH = '/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset'

# Option A: change working directory to parent folder of dataset
os.chdir(os.path.dirname(DRIVE_DATASET_PATH))
print('Working directory:', os.getcwd())

# Verify dataset folder exists
print('Dataset folder exists:', os.path.exists(DRIVE_DATASET_PATH))
```

If the boolean prints `False`, double-check the folder name in your Drive (caps / spacing). You can inspect `/content/drive/MyDrive/` with `!ls -la '/content/drive/MyDrive/'`.

---

### Cell 4 — Verify dataset structure and data.yaml (PYTHON CELL)

```python
from pathlib import Path
p = Path('/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset')
print('Exists:', p.exists())
print('List top-level:', list(p.iterdir())[:20])

# Print the data.yaml content
yaml_path = p / 'data.yaml'
if yaml_path.exists():
    print('\n--- data.yaml ---')
    print(yaml_path.read_text())
else:
    print('data.yaml not found at expected location:', yaml_path)
```

---

### Cell 5 — Update data.yaml paths for Colab (PYTHON CELL)

We will rewrite `data.yaml` so paths are correct for Colab. This also avoids problems with spaces by using the dataset root as `path:`.

```python
import textwrap
from pathlib import Path

root = Path('/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset')
new_yaml = textwrap.dedent(f"""
# YOLOv11 dataset configuration
path: {root}
train: images/train
val: images/val
test: images/test

# Classes
nc: 7
names: ['car-damage', 'crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
""")

# Save backup and update
backup = root / 'data.yaml.bak'
if (root / 'data.yaml').exists():
    (root / 'data.yaml').rename(backup)
    print('Backed up original data.yaml to', backup)

with open(root / 'data.yaml', 'w') as f:
    f.write(new_yaml)

print('Wrote new data.yaml to', root / 'data.yaml')
print('\n', new_yaml)
```

---

### Cell 6 — Quick test training (PYTHON CELL - 1 epoch)

This tests everything with a tiny run (1 epoch) and uses the `YOLO` Python API.

```python
from ultralytics import YOLO
import os

os.makedirs('models', exist_ok=True)
print('Starting short test training (1 epoch)')

# Use the yolo11n-seg checkpoint. If you don't have it locally, the ultralytics code will try to download the correct weights
model = YOLO('yolo11n-seg.pt')  # small/fast segmentation model

result = model.train(
    data=str('/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset/data.yaml'),
    epochs=1,
    imgsz=640,
    batch=8,   # reduce if OOM
    name='car_damage_test',
    project='models',
    device=0  # GPU device 0
)

print('Test training finished')
```

Notes:
- If you get CUDA OOM, reduce `batch` to 4 or 2, or reduce `imgsz` to 512.
- If Colab's GPU isn't available (`device=0` error), try `device='cuda:0'` or remove device to let Ultralytics choose.

---

### Cell 7 — Full training (PYTHON CELL)

Run the full experiment after the test completes and looks good.

```python
# Full training example; tune epochs and batch as needed
model = YOLO('yolo11n-seg.pt')
result = model.train(
    data=str('/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset/data.yaml'),
    epochs=100,
    imgsz=640,
    batch=16,            # lower if OOM
    name='car_damage_full',
    project='models',
    patience=20,
    device=0,
    workers=2,
    cache=True
)
```

Tips:
- Use `cache=True` to speed up dataloading, but it uses RAM/disk.
- If you get session timeouts, periodically copy important output (weights) to Drive.

---

### Cell 8 — Save best model to Drive and download (PYTHON CELL)

```python
import shutil
from pathlib import Path

results_dir = Path('models') / 'car_damage_full' / 'weights'
best = results_dir / 'best.pt'
print('best exists:', best.exists())

# Copy to a safe Drive folder
target = Path('/content/drive/MyDrive/Yolov11-seg data set') / 'training_results'
target.mkdir(parents=True, exist_ok=True)

if best.exists():
    shutil.copy(best, target / 'best_model.pt')
    print('Copied best model to', target / 'best_model.pt')
else:
    print('Best model not found; check models/ folder for outputs')
```

---

## 3) Common issues & fixes

- `zsh: command not found: yolo` locally: The `yolo` CLI is installed with `ultralytics`, but on some systems the CLI isn't added to PATH until a new shell; in Colab using `ultralytics` Python API (`YOLO`) is more reliable.
- GPU not available in Colab: Ensure Runtime → Change runtime type → GPU is set. Use `!nvidia-smi` to check.
- CUDA OOM: reduce `batch` or `imgsz`, or use `model.train(..., device='cuda:0')` and smaller batches.
- Dataset path not found: verify exact Drive path and watch for spaces or capitalization.

---

## 4) Short checklist before running

1. Upload `car_damage_yolo_dataset` to: `/MyDrive/Yolov11-seg data set/` (so the full path matches the guide)
2. Enable GPU in Colab
3. Run cells 1 → 8 in order
4. Keep notebooks running; periodically save results to Drive

---

## 5) Extra tips

- If you want to avoid spaces entirely, rename the Drive folder to `Yolov11-seg-data-set` and update the path in the notebook accordingly.
- If you prefer the CLI, after `pip install ultralytics`, you can run:

```bash
# CLI example (sometimes works in Colab)
!yolo segment train data='/content/drive/MyDrive/Yolov11-seg data set/car_damage_yolo_dataset/data.yaml' model=yolo11n-seg.pt epochs=100
```

- For long experiments, use Colab Pro/Pro+ for longer runtimes and better GPUs.

---

## Finished

You can now paste these cells into Google Colab and run training. If you want, I can also:

- Create a ready-to-open Colab notebook (.ipynb) with the same cells, or
- Update your existing `YOLOv11_Car_Damage_Training.ipynb` to use the exact path and safe handling of spaces.

Tell me which you'd prefer and I'll create it next.