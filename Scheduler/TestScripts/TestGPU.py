import torch

def check_gpu():
    print("Checking GPU availability...")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available.")
        print(f"[GPU] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Device Count: {torch.cuda.device_count()}")
        print(f"[INFO] Current CUDA Device: {torch.cuda.current_device()}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
        print(f"[INFO] PyTorch Version: {torch.__version__}")
    else:
        print("[WARNING] CUDA is not available. Using CPU.")
        print(f"[INFO] PyTorch Version: {torch.__version__}")
        print("[INFO] To enable GPU support, install PyTorch with CUDA support:")
        print("       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    check_gpu()
