import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from apv_config import Config
from model import VoxelMLP
from dataset import load_and_preprocess_data
import exr_util


def _safe_norm(idx: int, dim: int) -> float:
    """Normalize integer grid index to [0, 1]. Same as apv_train_dataset_maker._safe_norm."""
    if dim <= 1:
        return 0.0
    return float(idx) / float(dim - 1)


def _iter_chunks(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end


def evaluate_model(config: Config):
    """
    Main Evaluation Logic (APV):
    1. Load model and set to inference mode
    2. Iterate through samples and compute MSE (sum) + RMSE stats
    3. Generate predicted EXR slices along Y axis (match apv_train_dataset_maker.py)
    """
    # Step 1: Load Model
    print(f"\nLoading model: {config.model_path}")
    model = VoxelMLP(config).to(config.device)

    # Load Weights
    try:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load model! Error: {e}")
        return

    # Step 2: Set to Evaluation Mode
    model.eval()

    # Step 3: Initialize Loss Statistics
    total_l2_loss = 0.0
    l2_loss_fn = nn.MSELoss(reduction="sum")

    # Determine which dataloader to use
    dataloader = getattr(config, "dataloader", None)
    if dataloader is None:
        _, eval_loader, _ = load_and_preprocess_data(config)
        dataloader = eval_loader
        config.dataloader = dataloader

    if dataloader is None:
        print("No data to evaluate.")
        return

    config.total_samples = len(dataloader.dataset)

    print(f"\nStarting Inference Evaluation (Device: {config.device})...")

    # Step 4: Batch Inference + Loss Calculation
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (batch_inputs, batch_labels) in pbar:
            batch_inputs = batch_inputs.to(config.device, non_blocking=True)
            batch_labels = batch_labels.to(config.device, non_blocking=True)

            batch_preds = model(batch_inputs)
            batch_l2_loss = l2_loss_fn(batch_preds, batch_labels)
            total_l2_loss += batch_l2_loss.item()

            pbar.set_postfix({
                "Batch Avg L2 Loss": f"{batch_l2_loss.item() / batch_inputs.size(0):.4f}"
            })

    # Step 5: Calculate Average Loss
    avg_sample_l2_loss = total_l2_loss / config.total_samples
    print(f"\nEvaluation Results:")
    print(f"   Total Cumulative L2 Loss: {total_l2_loss:.4f}")
    print(f"   Total Samples: {config.total_samples}")
    print(f"   Avg Pixel L2 Loss per Sample: {avg_sample_l2_loss:.4f}")

    # Calculate RMSE
    avg_sample_rmse = np.sqrt(avg_sample_l2_loss)
    print(f"   Avg Pixel RMSE per Sample: {avg_sample_rmse:.4f}")

    # ===================== Generate Predicted EXR Slices (Y-sliced) =====================
    nx, ny, nz = (int(config.volume_dim[0]), int(config.volume_dim[1]), int(config.volume_dim[2]))
    chunk_size = int(getattr(config, "eval_batch_size", 131072))
    if chunk_size <= 0:
        chunk_size = 131072

    x_lin = torch.linspace(0.0, 1.0, steps=nx, device=config.device) if nx > 1 else torch.tensor([0.0], device=config.device)
    z_lin = torch.linspace(0.0, 1.0, steps=nz, device=config.device) if nz > 1 else torch.tensor([0.0], device=config.device)

    eval_dir = os.path.join(config.save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    with torch.no_grad():
        for iy in tqdm(range(ny), desc="Write pred EXR (slice=Y)"):
            y_norm = _safe_norm(iy, ny)

            zz, xx = torch.meshgrid(z_lin, x_lin, indexing="ij")  # (nz, nx)
            yy = torch.full((nz, nx), float(y_norm), device=config.device, dtype=torch.float32)
            coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (nz*nx, 3)

            out_cpu = torch.empty((coords.shape[0], 3), dtype=torch.float32, device="cpu")
            for start, end in _iter_chunks(coords.shape[0], chunk_size):
                preds = model(coords[start:end]).detach().cpu()
                out_cpu[start:end] = preds

            pred_zx = out_cpu.reshape(nz, nx, 3).numpy()  # (nz, nx, 3)
            # Match apv_train_dataset_maker.py: np.transpose(volume_yzx[iy], (1,0,2))
            slice_data = np.transpose(pred_zx, (1, 0, 2)) if getattr(config, "exr_xz_transpose", True) else pred_zx

            filename = f"pred_ambient_slice_{iy}.exr"
            path = os.path.join(eval_dir, filename)
            exr_util.write_exr(path, slice_data.astype(np.float32, copy=False))

    print(f"Saved predicted EXR slices to: {eval_dir}")


def main(config: Config):
    print("======= Voxel MLP Model Evaluation Script (APV) =======")
    print(f"Eval Config:")
    print(f"   Device: {config.device}")
    print(f"   Batch Size: {config.batch_size}")

    # Load Data (Use Full Dataset)
    _, eval_loader, _ = load_and_preprocess_data(config)
    config.dataloader = eval_loader
    config.total_samples = len(eval_loader.dataset)

    # Execute Evaluation
    evaluate_model(config)
    print("\nEvaluation Complete!")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, "data/APV_Bricks_L0_SampleScene.json")
    config = Config(json_path)
    main(config)

