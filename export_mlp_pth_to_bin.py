import argparse
import os
import json
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from apv_config import Config as APVConfig
from model import VoxelMLP


MAGIC = b"NGI_MLP\x00"  # 8 bytes
VERSION = 2


def _load_state_dict(pth_path: str, device: torch.device) -> dict:
    obj = torch.load(pth_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        # assume it's already a state_dict
        return obj
    raise TypeError(f"Unsupported .pth content type: {type(obj)}")


def _iter_linear_layers(model: nn.Module):
    # model.VoxelMLP keeps layers in model.model (nn.Sequential)
    seq = getattr(model, "model", None)
    if seq is None:
        raise AttributeError("Model has no attribute `model` (expected nn.Sequential)")
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            yield m


def _get_siren_omega0(model: nn.Module) -> float:
    """Extract omega_0 from the first SirenLayer in the model."""
    seq = getattr(model, "model", None)
    if seq is not None:
        for m in seq.modules():
            if hasattr(m, "w0"):
                return float(m.w0)
    return 30.0


def export_mlp_to_bin(pth_path: str, json_path: str, out_path: str) -> Tuple[int, int]:
    """
    Export VoxelMLP linear weights/biases to a binary file (v2).

    Binary layout (little-endian):
      ─── header ───
      - magic:          8 bytes  b'NGI_MLP\\0'
      - version:        uint32   2
      ─── metadata ───
      - omega_0:        float32  SIREN frequency
      - volume_dim:     3×uint32 [nx, ny, nz]
      - volume_origin:  3×float32 world-space origin
      - spacing:        float32  probe spacing
      ─── layers ───
      - num_layers:     uint32
      - for each layer:
          - in_features:  uint32
          - out_features: uint32
          - weights:      float32[out_features × in_features]  (row-major)
          - bias:         float32[out_features]
    """
    pth_path = os.path.abspath(pth_path)
    json_path = os.path.abspath(json_path)
    out_path = os.path.abspath(out_path)

    cfg = APVConfig(json_path)
    model = VoxelMLP(cfg).to(torch.device("cpu"))

    state_dict = _load_state_dict(pth_path, device=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    layers = list(_iter_linear_layers(model))
    if len(layers) == 0:
        raise RuntimeError("No nn.Linear layers found in model.")

    omega_0 = _get_siren_omega0(model)
    nx, ny, nz = cfg.volume_dim
    ox, oy, oz = cfg.apv_origin
    spacing = cfg.apv_spacing

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        # header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))

        # metadata
        f.write(struct.pack("<f", omega_0))
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<fff", ox, oy, oz))
        f.write(struct.pack("<f", spacing))

        # layers
        f.write(struct.pack("<I", len(layers)))
        for layer in layers:
            w = layer.weight.detach().cpu().numpy().astype(np.float32, copy=False)
            b = layer.bias.detach().cpu().numpy().astype(np.float32, copy=False)

            out_features, in_features = w.shape
            f.write(struct.pack("<I", int(in_features)))
            f.write(struct.pack("<I", int(out_features)))
            f.write(w.tobytes(order="C"))
            f.write(b.tobytes(order="C"))

    print(f"  omega_0     = {omega_0}")
    print(f"  volume_dim  = ({nx}, {ny}, {nz})")
    print(f"  origin      = ({ox}, {oy}, {oz})")
    print(f"  spacing     = {spacing}")

    total_params = sum(int(layer.weight.numel() + layer.bias.numel()) for layer in layers)
    return len(layers), total_params


def export_mlp_to_json(pth_path: str, json_path: str, out_json_path: str) -> Tuple[int, int]:
    """
    Export the same ordered structure as the .bin v2 into a JSON file for debugging/comparison.
    """
    pth_path = os.path.abspath(pth_path)
    json_path = os.path.abspath(json_path)
    out_json_path = os.path.abspath(out_json_path)

    cfg = APVConfig(json_path)
    model = VoxelMLP(cfg).to(torch.device("cpu"))

    state_dict = _load_state_dict(pth_path, device=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    layers = list(_iter_linear_layers(model))
    if len(layers) == 0:
        raise RuntimeError("No nn.Linear layers found in model.")

    omega_0 = _get_siren_omega0(model)
    nx, ny, nz = cfg.volume_dim
    ox, oy, oz = cfg.apv_origin

    payload = {
        "magic": MAGIC.decode("latin1"),
        "version": VERSION,
        "omega_0": omega_0,
        "volume_dim": [nx, ny, nz],
        "volume_origin": [ox, oy, oz],
        "spacing": cfg.apv_spacing,
        "num_layers": len(layers),
        "layers": [],
    }

    for layer in layers:
        w = layer.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        b = layer.bias.detach().cpu().numpy().astype(np.float32, copy=False)
        out_features, in_features = w.shape
        payload["layers"].append(
            {
                "in_features": int(in_features),
                "out_features": int(out_features),
                "weight": w.tolist(),
                "bias": b.tolist(),
            }
        )

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    total_params = sum(int(layer.weight.numel() + layer.bias.numel()) for layer in layers)
    return len(layers), total_params


def main():
    current_dir = Path(__file__).resolve().parent
    default_json = os.path.join(current_dir, "data", "APV_Bricks_L0_SampleScene.json")
    default_pth = os.path.join(current_dir, "apv_model_checkpoints", "mlp_final.pth")
    default_out = os.path.join(current_dir, "apv_model_checkpoints", "mlp_final.bin")
    default_out_json = os.path.join(current_dir, "apv_model_checkpoints", "mlp_final.json")

    p = argparse.ArgumentParser(description="Export VoxelMLP .pth weights/biases to a compact binary file.")
    p.add_argument("--json", dest="json_path", type=str, default=default_json, help="Path to APV JSON (for model config)")
    p.add_argument("--pth", dest="pth_path", type=str, default=default_pth, help="Path to .pth (state_dict or checkpoint dict)")
    p.add_argument("--out", dest="out_path", type=str, default=default_out, help="Output .bin path")
    p.add_argument("--out-json", dest="out_json_path", type=str, default=default_out_json, help="Output .json path (debug/compare)")
    args = p.parse_args()

    n_layers, n_params = export_mlp_to_bin(args.pth_path, args.json_path, args.out_path)
    print(f"Exported {n_layers} Linear layers, total params={n_params:,}")
    print(f"Saved: {os.path.abspath(args.out_path)}")

    j_layers, j_params = export_mlp_to_json(args.pth_path, args.json_path, args.out_json_path)
    print(f"Exported JSON {j_layers} Linear layers, total params={j_params:,}")
    print(f"Saved: {os.path.abspath(args.out_json_path)}")


if __name__ == "__main__":
    main()

