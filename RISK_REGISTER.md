# Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Kaggle folder name differs after unzip | Scripts fail to find raw data | Inspect extracted folder and update config |
| `valid/` vs `val/` mismatch | Validation path error | Map raw `valid` to processed `val` |
| Tomato class names differ slightly | Missing classes | Print all folders and verify exact names |
| Class imbalance | Accuracy misleading | Use class weights and macro F1 |
| Corrupted test accidentally used in training | Invalid robustness results | Keep corrupted set under separate folder and evaluate only |
| Too many models | Timeline risk | Train required 4 models first; MobileViT-XS optional |
| Grad-CAM incompatible with MobileViT target layer | Visualization issue | Use appropriate convolutional feature layer; keep MobileViT exploratory |
| Latency results inconsistent | Poor benchmark quality | Average over at least 100 images, log hardware |
| Overengineering demo/app | Research pipeline delayed | Build Streamlit demo only after experiments finish |
| Claiming Grad-CAM proves causality | Academic weakness | State Grad-CAM is post-hoc and qualitative |
