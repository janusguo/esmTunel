import torch
from transformers import AutoModel
import os

def test_load():
    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Attempting to load {model_name}...")
    try:
        # Load in float16 to save memory
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        print("Model loaded in float16")
        
        # Check memory
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage: {mem:.2f} MB")
        
        # Try a dummy forward pass
        input_ids = torch.randint(0, 20, (1, 512))
        with torch.no_grad():
            outputs = model(input_ids)
        print("Forward pass successful")
        
        mem = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage after forward: {mem:.2f} MB")
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_load()
