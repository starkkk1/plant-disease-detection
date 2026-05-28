# Tech Stack

## Core Stack

| Component | Choice |
|---|---|
| Language | Python 3.10+ |
| Framework | PyTorch |
| Models | torchvision + timm |
| Metrics | scikit-learn + torchmetrics |
| Image processing | Pillow + OpenCV |
| Visualization | matplotlib |
| Grad-CAM | pytorch-grad-cam |
| Config | PyYAML |
| Optional demo | Streamlit |
| Optional deployment benchmark | ONNX Runtime |

## Required Libraries

```text
torch
torchvision
timm
numpy
pandas
matplotlib
scikit-learn
torchmetrics
opencv-python
pillow
tqdm
PyYAML
pytorch-grad-cam
```

## Optional Libraries

```text
onnx
onnxruntime
streamlit
ptflops
thop
```

## Hardware

Minimum:

- Google Colab GPU or local CPU for small tests.

Recommended:

- NVIDIA GPU with CUDA.

If CPU only:

- reduce batch size to 8 or 16,
- start with MobileNetV3-Small,
- avoid training MobileViT-XS until main pipeline works.
