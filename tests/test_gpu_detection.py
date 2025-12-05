import torch


def test_gpu_detection():
    """Check if GPUs are detected."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs detected: {gpu_count}")
    else:
        print("No GPUs detected.")


if __name__ == "__main__":
    test_gpu_detection()
