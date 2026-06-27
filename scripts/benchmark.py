import os
import time
import torch
import timm

def get_file_size_mb(filepath):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_path = os.path.join(base_dir, filepath)
    if os.path.exists(full_path):
        return os.path.getsize(full_path) / (1024 * 1024)
    return 0

def benchmark_model(model_name, num_classes=11, device='cpu', batch_size=1):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    
    # Warmup
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    num_iterations = 1000
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    latency_ms = ((end_time - start_time) / num_iterations) * 1000 / batch_size
    return params, latency_ms

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on device: {device}")
    
    models = [
        ('ConvNeXt Tiny (Teacher)', 'convnext_tiny', 'checkpoints/convnext_tiny_best.pth'),
        ('EfficientNetB0 (Student)', 'efficientnet_b0', 'checkpoints/efficientnet_b0_distilled_best.pth'),
        ('MobileNetV3 Small (Student)', 'mobilenetv3_small_100', 'checkpoints/mobilenetv3_small_100_distilled_best.pth')
    ]
    
    print("-" * 80)
    print(f"{'Model':<30} | {'Params (M)':<12} | {'Size (MB)':<10} | {'Latency (ms)':<15}")
    print("-" * 80)
    
    for display_name, timm_name, ckpt_path in models:
        try:
            params, latency = benchmark_model(timm_name, device=device)
            size_mb = get_file_size_mb(ckpt_path)
            print(f"{display_name:<30} | {params/1e6:>10.2f} M | {size_mb:>8.2f} MB | {latency:>10.2f} ms")
        except Exception as e:
            print(f"Failed {timm_name}: {e}")
    print("-" * 80)
