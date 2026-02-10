import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class Config:
    """
    APV 专用 Config：
    - 从 Unity APV 导出的 JSON（minBrickSize + bricks/probes）推导 volume_dim
    - volume_dim 的计算逻辑与 apv_train_dataset_maker.py 保持一致
    - checkpoint 输出目录与原版分开（apv_model_checkpoints/）
    """

    def __init__(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        if "minBrickSize" not in data:
            raise KeyError("JSON missing required key: minBrickSize")
        if "bricks" not in data:
            raise KeyError("JSON missing required key: bricks")

        min_brick_size = float(data["minBrickSize"])
        spacing = min_brick_size / 3.0
        if spacing <= 0:
            raise ValueError(f"Invalid spacing computed from minBrickSize: {min_brick_size}")

        # Gather probes (same filter rule as apv_train_dataset_maker.py to keep dims consistent)
        pos_list: List[Tuple[float, float, float]] = []
        for brick in data["bricks"]:
            probes = brick.get("probes", [])
            for probe in probes:
                pos = probe.get("posWS", None)
                l0 = probe.get("L0", None)
                if pos is None or l0 is None:
                    continue
                if len(pos) != 3 or len(l0) != 3:
                    continue
                pos_list.append((float(pos[0]), float(pos[1]), float(pos[2])))

        if len(pos_list) == 0:
            raise ValueError("No probes found in JSON (expected bricks[*].probes[*].posWS and L0).")

        pos_np = np.asarray(pos_list, dtype=np.float64)  # (N, 3) in XYZ
        origin = pos_np.min(axis=0)

        # Map positions -> integer grid indices using rint, then shift to start from 0
        idx_np = np.rint((pos_np - origin) / spacing).astype(np.int64)
        idx_np -= idx_np.min(axis=0)

        dims_xyz = (idx_np.max(axis=0) + 1).astype(np.int64)
        self.volume_dim = [int(dims_xyz[0]), int(dims_xyz[1]), int(dims_xyz[2])]  # [nx, ny, nz]

        # Keep for debugging (not used by training)
        self.apv_origin = origin.tolist()
        self.apv_spacing = float(spacing)

        current_dir = Path(__file__).resolve().parent

        # ===================== Data Settings =====================
        self.data_path = os.path.join(current_dir, "data/train.npy")  # Path to training data

        # ===================== Model Settings =====================
        self.input_dim = 3  # XYZ coordinates
        self.hidden_dims = [64, 64, 64, 64, 64, 64]  # MLP hidden layers
        self.output_dim = 3  # RGB labels
        self.activation = nn.ReLU()

        # ===================== Training Settings =====================
        self.batch_size = 512
        self.lr = 1e-3
        self.epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===================== Eval / Export Settings =====================
        # EXR 约定：按 Y 切片（与 apv_train_dataset_maker.py 一致），每张是 XZ 平面，并做同样的 transpose
        self.slice_axis = "y"
        self.exr_xz_transpose = True
        # 推理时的 chunk 大小（避免一次性推理过多点导致显存爆）
        self.eval_batch_size = 131072

        # ===================== Saving Settings =====================
        self.save_dir = os.path.join(current_dir, "apv_model_checkpoints")
        self.model_path = os.path.join(self.save_dir, "mlp_final.pth")
        self.save_freq = 100
        self.plot_freq = 100

