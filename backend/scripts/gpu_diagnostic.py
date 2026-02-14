import torch


def test_gpu():
    print("--- GPU DIAGNOSTIC ---")
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Check your drivers.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"✅ Detected GPU: {gpu_name}")
    print(f"✅ Total VRAM: {total_vram:.2f} GB")

    try:
        # Try to allocate a small tensor (100MB) to test memory access
        print("🔄 Attempting memory allocation test...")
        x = torch.ones((1024, 1024, 25), device="cuda")
        print("✅ Memory allocation successful!")
        print("--- DIAGNOSTIC PASSED ---")
    except Exception as e:
        print(f"❌ Memory allocation FAILED: {e}")


if __name__ == "__main__":
    test_gpu()