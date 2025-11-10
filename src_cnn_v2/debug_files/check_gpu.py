import sys

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 30)

    if torch.cuda.is_available():
        print("✅ SUCCESS: PyTorch found a CUDA-enabled GPU.")
        print(f"   - GPU Device Count: {torch.cuda.device_count()}")
        print(f"   - Current GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ FAILED: PyTorch was found, but CUDA is not available.")
        print("   This means your PyTorch installation is CPU-only or the CUDA driver is incompatible.")

except ImportError:
    print("❌ FAILED: PyTorch is not installed in this environment.")
    sys.exit(1)

# python src_cnn_v2/debug_files/check_gpu.py