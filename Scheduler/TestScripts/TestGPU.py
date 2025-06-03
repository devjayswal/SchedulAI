import torch

def check_gpu():
    print("Checking GPU availability...")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA is available.")
        print(f"🖥️  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"🚀 CUDA Device Count: {torch.cuda.device_count()}")
        print(f"🧠 Current CUDA Device: {torch.cuda.current_device()}")
    else:
        print("❌ CUDA is not available. Using CPU.")

if __name__ == "__main__":
    check_gpu()
