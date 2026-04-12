import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

def load_and_preprocess_data(config, data_path: Optional[str] = None):
    # Load data
    if data_path is None:
        if not hasattr(config, "data_path"):
            raise ValueError("Config missing data_path")
        data_path = config.data_path

    print(f"Loading data: {data_path}")
        
    try:
        data = np.load(data_path).astype(np.float32)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        return None, None, 0

    print(f"Original data shape: {data.shape} (Samples x 6)")
    
    # Split inputs and labels
    inputs = data[:, :3]   # First 3 columns: XYZ coordinates
    labels = data[:, 3:]   # Last 3 columns: RGB values
    
    # Keep dataset tensors on CPU; move to GPU per-batch in the training/eval loops.
    inputs_tensor = torch.from_numpy(inputs)
    labels_tensor = torch.from_numpy(labels)
    
    total_samples = len(inputs)
    
    # Create DataLoaders
    train_dataset = TensorDataset(inputs_tensor, labels_tensor)

    device = getattr(config, "device", torch.device("cpu"))
    pin_memory = getattr(device, "type", "cpu") == "cuda"
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,  # Shuffle training data
        num_workers=0,  # Set to 0 for Windows to avoid multi-threading issues
        pin_memory=pin_memory,
    )
    
    # Loader for evaluation (using full dataset, no shuffle)
    eval_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False, # Don't shuffle for evaluation
        num_workers=0,
        pin_memory=pin_memory,
    )
    
    print(f"Data preprocessing complete:")
    print(f"   Total samples: {total_samples}")
    print(f"   Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"   Label range: [{labels.min():.4f}, {labels.max():.4f}]")
    
    # Return train_loader (shuffled) and eval_loader (not shuffled)
    return train_loader, eval_loader, total_samples
