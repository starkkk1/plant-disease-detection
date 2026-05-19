# Tech Stack

This project uses PyTorch. Do not switch frameworks mid-project.

---

## Core Stack

| Component      | Choice                     | Reason                                   |
|----------------|----------------------------|------------------------------------------|
| Language       | Python 3.10+               | Standard for DL research                 |
| Framework      | PyTorch                    | Flexible, supports custom Grad-CAM       |
| Models         | torchvision + timm         | Pretrained weights, standard API         |
| Grad-CAM       | pytorch-grad-cam           | Clean library, no reinventing the wheel  |
| Metrics        | scikit-learn + torchmetrics| Confusion matrix, F1 macro               |
| Config         | PyYAML                     | Human-readable, no hardcoded values      |
| Visualization  | matplotlib                 | Plots, heatmaps, confusion matrices      |
| Image handling | Pillow, OpenCV             | Load and overlay Grad-CAM                |

---

## requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pytorch-grad-cam>=1.4.0
scikit-learn>=1.3.0
torchmetrics>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
opencv-python>=4.8.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.65.0
```

Optional — experiment tracking:

```
tensorboard>=2.14.0
```

Optional — demo:

```
streamlit>=1.28.0
```

---

## Hardware

| Tier       | Setup                                      |
|------------|--------------------------------------------|
| Minimum    | Google Colab (free T4 GPU) — acceptable    |
| Recommended| Local NVIDIA GPU, CUDA 11.8+, 8GB+ VRAM   |

If training on CPU only: reduce batch size to 8 and expect ~10× slower training.

---

## Seed / Reproducibility

```python
# src/utils/seed.py
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Call `set_seed(42)` at the top of every training and evaluation script.

---

## Config System

All hyperparameters live in `configs/`. Training scripts read from config only.

```yaml
# configs/default.yaml (excerpt)
seed: 42
data:
  root: data/processed/tomato
  image_size: 224
  batch_size: 32
  num_workers: 4
training:
  epochs: 30
  lr: 1.0e-4
  fine_tune_lr: 1.0e-5
  patience: 5
  optimizer: adam
model:
  name: mobilenet_v2        # override in model-specific config
  num_classes: 10
  pretrained: true
output:
  checkpoint_dir: checkpoints
  results_dir: results
```

Model-specific configs inherit default and override `model.name` only (unless different LR needed).
