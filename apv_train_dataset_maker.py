import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np

import exr_util


def _safe_norm(idx: int, dim: int) -> float:
    """Normalize integer grid index to [0, 1]."""
    if dim <= 1:
        return 0.0
    return float(idx) / float(dim - 1)


def _build_dataset_from_probes(
    pos_list: List[List[float]],
    rgb_list: List[List[float]],
    origin: np.ndarray,
    spacing: float,
    output_dir: str,
    *,
    scale_factor: float = 1.0,
    save_exr: bool = True,
    max_exr_slices: Optional[int] = None,
    label: str = "",
) -> None:
    """
    Core logic: given collected probe positions and RGB values, build and save:
      - train.npy  (rows: [x_norm, y_norm, z_norm, r, g, b])
      - textures/ambient_slice_{iy}.exr  (XZ-plane slices along Y)

    Args:
        pos_list:  list of [x, y, z] world-space positions
        rgb_list:  list of [r, g, b] L0 values
        origin:    world-space origin (component-wise min of all positions in the
                   GLOBAL dataset; pass per-cell min to make each cell self-contained)
        spacing:   uniform probe spacing (minBrickSize / 3)
        output_dir: directory to write train.npy and textures/ into
        label:     string prefix for log messages (e.g. "cell 3")
    """
    os.makedirs(output_dir, exist_ok=True)

    pos_np = np.asarray(pos_list, dtype=np.float64)  # (N, 3)
    rgb_np = np.asarray(rgb_list, dtype=np.float32)  # (N, 3)

    # Map world positions to integer grid indices
    idx_np = np.rint((pos_np - origin) / spacing).astype(np.int64)
    idx_np -= idx_np.min(axis=0)

    dims_xyz = (idx_np.max(axis=0) + 1).astype(np.int64)
    nx, ny, nz = int(dims_xyz[0]), int(dims_xyz[1]), int(dims_xyz[2])
    total_voxels = nx * ny * nz

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Origin: {origin.tolist()}")
    print(f"{prefix}Spacing: {spacing}")
    print(f"{prefix}Grid dims (Nx, Ny, Nz): {(nx, ny, nz)} | total={total_voxels:,}")

    # Build dense volume [y, z, x, 3] for convenient Y-axis slicing
    volume_yzx = np.zeros((ny, nz, nx, 3), dtype=np.float32)

    seen: Dict[Tuple[int, int, int], int] = {}
    dup_count = 0
    filled = 0
    for i in range(idx_np.shape[0]):
        ix, iy, iz = int(idx_np[i, 0]), int(idx_np[i, 1]), int(idx_np[i, 2])
        key = (ix, iy, iz)
        if key in seen:
            dup_count += 1
            continue
        seen[key] = i
        volume_yzx[iy, iz, ix] = rgb_np[i]
        filled += 1

    print(f"{prefix}Probes: {idx_np.shape[0]} | unique: {filled} | duplicates dropped: {dup_count}")

    # Sanity check
    recon_pos = origin + idx_np.astype(np.float64) * spacing
    max_abs_err = np.max(np.abs(recon_pos - pos_np))
    print(f"{prefix}Max |posWS - (origin + idx*spacing)| = {max_abs_err:.6g}")

    # Build flat dataset rows: [x_norm, y_norm, z_norm, r, g, b]
    dataset = np.zeros((total_voxels, 6), dtype=np.float32)
    out_min = out_max = None
    k = 0
    for iy in range(ny):
        y_norm = _safe_norm(iy, ny)
        for iz in range(nz):
            z_norm = _safe_norm(iz, nz)
            for ix in range(nx):
                x_norm = _safe_norm(ix, nx)
                r, g, b = (volume_yzx[iy, iz, ix] / float(scale_factor)).tolist()
                dataset[k, :] = (x_norm, y_norm, z_norm, r, g, b)
                if out_min is None:
                    out_min, out_max = min(r, g, b), max(r, g, b)
                else:
                    out_min = min(out_min, r, g, b)
                    out_max = max(out_max, r, g, b)
                k += 1

    print(f"{prefix}RGB range (after /scale_factor): min={out_min}, max={out_max}")

    saved_path = os.path.join(output_dir, "train.npy")
    np.save(saved_path, dataset)
    print(f"{prefix}Saved {saved_path}, shape={dataset.shape}")

    if save_exr:
        tex_dir = os.path.join(output_dir, "textures")
        os.makedirs(tex_dir, exist_ok=True)

        num_slices = ny if max_exr_slices is None else min(ny, int(max_exr_slices))
        for iy in range(num_slices):
            # Transpose to (width=Nx, height=Nz, 3) for EXR convention
            slice_data = np.transpose(volume_yzx[iy], (1, 0, 2))
            path = os.path.join(tex_dir, f"ambient_slice_{iy}.exr")
            exr_util.write_exr(path, slice_data)
        print(f"{prefix}Saved {num_slices} EXR slices to {tex_dir}")


# ---------------------------------------------------------------------------
# Public API – all probes combined
# ---------------------------------------------------------------------------

def make_apv_train_dataset(
    json_path: str,
    output_dir: str,
    *,
    scale_factor: float = 1.0,
    save_exr: bool = True,
    max_exr_slices: Optional[int] = None,
) -> None:
    """
    Build a single training dataset from all probes in the APV JSON.

    Output layout:
        output_dir/
          train.npy
          textures/
            ambient_slice_0.exr
            ...
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading APV JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _check_required_keys(data)
    spacing = float(data["minBrickSize"]) / 3.0

    pos_list, rgb_list = _gather_probes(data["bricks"])
    if not pos_list:
        raise ValueError("No probes found in JSON.")

    origin = np.asarray(pos_list, dtype=np.float64).min(axis=0)
    _build_dataset_from_probes(
        pos_list, rgb_list, origin, spacing, output_dir,
        scale_factor=scale_factor, save_exr=save_exr,
        max_exr_slices=max_exr_slices,
    )


# ---------------------------------------------------------------------------
# Public API – one dataset per cell
# ---------------------------------------------------------------------------

def make_apv_train_dataset_per_cell(
    json_path: str,
    output_dir: str,
    *,
    scale_factor: float = 1.0,
    save_exr: bool = True,
    max_exr_slices: Optional[int] = None,
) -> None:
    """
    Build one training dataset per cellIndex found in the APV JSON.

    Output layout:
        output_dir/
          cell_0/
            train.npy
            textures/
              ambient_slice_0.exr  ...
          cell_1/
            ...
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading APV JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _check_required_keys(data)
    spacing = float(data["minBrickSize"]) / 3.0

    # Group bricks by cellIndex
    cell_bricks: Dict[int, list] = defaultdict(list)
    for brick in data["bricks"]:
        cell_idx = brick.get("cellIndex")
        if cell_idx is None:
            continue
        cell_bricks[int(cell_idx)].append(brick)

    cell_ids = sorted(cell_bricks.keys())
    print(f"Found {len(cell_ids)} cells: {cell_ids}")

    for cell_id in cell_ids:
        bricks = cell_bricks[cell_id]
        pos_list, rgb_list = _gather_probes(bricks)
        if not pos_list:
            print(f"[cell {cell_id}] No valid probes, skipping.")
            continue

        cell_origin = np.asarray(pos_list, dtype=np.float64).min(axis=0)
        cell_out_dir = os.path.join(output_dir, f"cell_{cell_id}")

        print(f"\n--- Cell {cell_id} ({len(bricks)} bricks, {len(pos_list)} probes) ---")
        _build_dataset_from_probes(
            pos_list, rgb_list, cell_origin, spacing, cell_out_dir,
            scale_factor=scale_factor, save_exr=save_exr,
            max_exr_slices=max_exr_slices,
            label=f"cell {cell_id}",
        )

    print(f"\nDone. Per-cell datasets saved under: {output_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required_keys(data: dict) -> None:
    for key in ("minBrickSize", "bricks"):
        if key not in data:
            raise KeyError(f"JSON missing required key: {key}")
    spacing = float(data["minBrickSize"]) / 3.0
    if spacing <= 0:
        raise ValueError(f"Invalid spacing from minBrickSize={data['minBrickSize']}")


def _gather_probes(
    bricks: list,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Extract posWS and L0 from a list of brick dicts."""
    pos_list: List[List[float]] = []
    rgb_list: List[List[float]] = []
    for brick in bricks:
        for probe in brick.get("probes", []):
            pos = probe.get("posWS")
            l0 = probe.get("L0")
            if pos is None or l0 is None:
                continue
            if len(pos) != 3 or len(l0) != 3:
                continue
            pos_list.append([float(pos[0]), float(pos[1]), float(pos[2])])
            rgb_list.append([float(l0[0]), float(l0[1]), float(l0[2])])
    return pos_list, rgb_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser(default_json: str, default_out: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build train.npy + y-sliced EXR textures from Unity APV probe JSON."
    )
    p.add_argument("--json", dest="json_path", type=str, default=default_json,
                   help="Path to APV exported JSON")
    p.add_argument("--out", dest="output_dir", type=str, default=default_out,
                   help="Output directory")
    p.add_argument("--scale-factor", dest="scale_factor", type=float, default=1.0,
                   help="Divide RGB by this factor in train.npy")
    p.add_argument("--no-exr", dest="save_exr", action="store_false",
                   help="Disable EXR export")
    p.add_argument("--max-exr-slices", dest="max_exr_slices", type=int, default=None,
                   help="Limit EXR slices per cell (for debugging)")
    p.add_argument("--per-cell", dest="per_cell", action="store_true",
                   help="Split output by cellIndex (one sub-folder per cell)")
    p.set_defaults(save_exr=True, per_cell=True)
    return p


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    default_json = os.path.join(
        current_dir, "data", "APV_Bricks_L0_TerminalScene_20260326_220506.json"
    )
    default_out = os.path.join(current_dir, "data")

    parser = _build_arg_parser(default_json, default_out)
    args = parser.parse_args()

    kwargs = dict(
        scale_factor=args.scale_factor,
        save_exr=args.save_exr,
        max_exr_slices=args.max_exr_slices,
    )

    if args.per_cell:
        make_apv_train_dataset_per_cell(args.json_path, args.output_dir, **kwargs)
    else:
        make_apv_train_dataset(args.json_path, args.output_dir, **kwargs)
