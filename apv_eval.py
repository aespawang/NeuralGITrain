import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from apv_config import Config
from model import get_model
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
    model = get_model(config.model, config).to(config.device)

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
    observed_gt_max = 0.0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (batch_inputs, batch_labels) in pbar:
            batch_inputs = batch_inputs.to(config.device, non_blocking=True)
            batch_labels = batch_labels.to(config.device, non_blocking=True)

            batch_preds = model(batch_inputs)
            batch_l2_loss = l2_loss_fn(batch_preds, batch_labels)
            total_l2_loss += batch_l2_loss.item()

            if batch_labels.numel() > 0:
                observed_gt_max = max(observed_gt_max, float(batch_labels.max().item()))

            pbar.set_postfix({
                "Batch Avg L2 Loss": f"{batch_l2_loss.item() / batch_inputs.size(0):.4f}"
            })

    # Step 5: Calculate Average Loss
    avg_sample_l2_loss = total_l2_loss / config.total_samples
    print(f"\nEvaluation Results:")
    print(f"   Total Cumulative L2 Loss: {total_l2_loss:.4f}")
    print(f"   Total Samples: {config.total_samples}")
    print(f"   Avg Pixel L2 Loss per Sample: {avg_sample_l2_loss:.4f}")

    # Calculate RMSE and PSNR (auto peak from GT max, override via config.psnr_max_val if provided)
    avg_sample_rmse = np.sqrt(avg_sample_l2_loss)
    cfg_peak = getattr(config, "psnr_max_val", None)
    max_val = float(cfg_peak) if cfg_peak is not None else (observed_gt_max if observed_gt_max > 0 else 1.0)
    psnr = float("inf") if avg_sample_l2_loss <= 0 else 10.0 * np.log10((max_val * max_val) / avg_sample_l2_loss)
    print(f"   Avg Pixel RMSE per Sample: {avg_sample_rmse:.4f}")
    print(f"   PSNR (peak={max_val}): {psnr:.4f} dB")

    # Prepare GT volume for SSIM (reshape back to [ny, nz, nx, 3])
    gt_volume_yzx = None
    try:
        gt_np = np.load(config.data_path).astype(np.float32)
        nx, ny, nz = int(config.volume_dim[0]), int(config.volume_dim[1]), int(config.volume_dim[2])
        expected = nx * ny * nz
        if gt_np.shape[0] >= expected:
            gt_labels = gt_np[:expected, 3:]
            gt_volume_yzx = gt_labels.reshape((ny, nz, nx, 3))
        else:
            print(f"Warning: GT data rows ({gt_np.shape[0]}) < expected voxels ({expected}); skip SSIM")
    except Exception as e:
        print(f"Warning: failed to load GT for SSIM: {e}")

    # ===================== Generate Predicted EXR Slices (Y-sliced) =====================
    nx, ny, nz = (int(config.volume_dim[0]), int(config.volume_dim[1]), int(config.volume_dim[2]))
    chunk_size = int(getattr(config, "eval_batch_size", 131072))
    if chunk_size <= 0:
        chunk_size = 131072

    x_lin = torch.linspace(0.0, 1.0, steps=nx, device=config.device) if nx > 1 else torch.tensor([0.0], device=config.device)
    z_lin = torch.linspace(0.0, 1.0, steps=nz, device=config.device) if nz > 1 else torch.tensor([0.0], device=config.device)

    eval_dir = os.path.join(config.save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    ssim_total = 0.0
    ssim_count = 0

    def _ssim_slice(pred_zx_np: np.ndarray, gt_zx_np: np.ndarray, peak: float) -> float:
        # pred_zx_np / gt_zx_np: (nz, nx, 3), float32
        pred = torch.from_numpy(pred_zx_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,nz,nx)
        gt = torch.from_numpy(gt_zx_np).permute(2, 0, 1).unsqueeze(0)
        C1 = (0.01 * peak) ** 2
        C2 = (0.03 * peak) ** 2
        mu_pred = F.avg_pool2d(pred, 3, 1, padding=1)
        mu_gt = F.avg_pool2d(gt, 3, 1, padding=1)
        sigma_pred = F.avg_pool2d(pred * pred, 3, 1, padding=1) - mu_pred * mu_pred
        sigma_gt = F.avg_pool2d(gt * gt, 3, 1, padding=1) - mu_gt * mu_gt
        sigma_cross = F.avg_pool2d(pred * gt, 3, 1, padding=1) - mu_pred * mu_gt
        ssim_map = ((2 * mu_pred * mu_gt + C1) * (2 * sigma_cross + C2)) / (
            (mu_pred * mu_pred + mu_gt * mu_gt + C1) * (sigma_pred + sigma_gt + C2)
        )
        return float(ssim_map.mean().item())

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

            # Compute SSIM for this slice if GT available
            if gt_volume_yzx is not None:
                gt_slice_zx = gt_volume_yzx[iy]  # (nz, nx, 3)
                ssim_total += _ssim_slice(pred_zx, gt_slice_zx, max_val)
                ssim_count += 1

    if ssim_count > 0:
        print(f"   SSIM (mean over {ssim_count} Y-slices): {ssim_total / ssim_count:.4f}")
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
    json_path = os.path.join(current_dir, "data/APV_Bricks_L0_Probe URP_20260325_185736.json")
    config = Config(json_path)
    main(config)

