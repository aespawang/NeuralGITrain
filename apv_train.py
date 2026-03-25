import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from apv_config import Config
from model import get_model
from dataset import load_and_preprocess_data
import apv_eval as eval
import time


# ===================== Training & Validation Functions =====================
def train_one_epoch(model, train_loader, criterion, optimizer, config, epoch, writer=None, global_step=0):
    """Train for one epoch."""
    model.train()  # Set to training mode
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

    for batch_inputs, batch_labels in pbar:
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * batch_inputs.size(0)
        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    # Calculate average loss
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss, global_step


# ===================== Visualization =====================
def plot_loss(train_losses, config, epoch):
    """Plot Training Loss Curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (Epoch {epoch+1})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(config.save_dir, f"loss_curve_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved: {save_path}")


# ===================== Main Training Flow =====================
def main(config: Config):
    print(f"Training Config (APV):")
    print(f"   Device: {config.device}")
    print(f"   Model Arch: {config.input_dim} -> {config.hidden_dims} -> {config.output_dim}")
    print(f"   Epochs: {config.epochs} | Batch Size: {config.batch_size} | LR: {config.lr}")
    print(f"   Volume Dim (Nx,Ny,Nz): {tuple(config.volume_dim)}")

    # Create model save directory
    os.makedirs(config.save_dir, exist_ok=True)
    tb_log_dir = os.path.join(config.save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Load Data
    train_loader, _, _ = load_and_preprocess_data(config)
    if train_loader is None:
        return

    # Initialize Model, Criterion, Optimizer
    model = get_model(config.model, config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Track Losses
    train_losses = []
    global_step = 0

    # Start Training
    print("\nStarting Training on FULL Dataset (APV)...")
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        train_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch, writer, global_step)
        train_losses.append(train_loss)
        scheduler.step()

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

        # Save Model
        if (epoch + 1) % config.save_freq == 0:
            model_path = os.path.join(config.save_dir, f"mlp_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
            }, model_path)
            print(f"Model saved: {model_path}")

        # Plot Loss
        if (epoch + 1) % config.plot_freq == 0:
            plot_loss(train_losses, config, epoch)

    end_time = time.perf_counter()
    total_seconds = end_time - start_time

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print(f"训练完成！总共耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")

    # Training Complete: Save Final Model
    final_model_path = os.path.join(config.save_dir, "mlp_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining Complete! Final model saved to: {final_model_path}")

    # Plot Final Loss Curve
    plot_loss(train_losses, config, config.epochs - 1)
    writer.close()


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, "data/APV_Bricks_L0_Probe URP_20260325_185736.json")
    config = Config(json_path)
    main(config)
    # Run evaluation after training
    eval.main(config)

