import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

import numpy as np

import exr_util


def _safe_norm(idx: int, dim: int) -> float:
    """Normalize integer grid index to [0, 1]."""
    if dim <= 1:
        return 0.0
    return float(idx) / float(dim - 1)


def make_apv_train_dataset(
    json_path: str,
    output_dir: str,
    *,
    scale_factor: float = 1.0,
    save_exr: bool = True,
    max_exr_slices: Optional[int] = None,
) -> None:
    """
    Build a training dataset from Unity APV exported JSON:
    - Use XZ as ground plane, Y up (Unity coordinate system).
    - Use probe `posWS` as world-space XYZ.
    - Use component-wise min(posWS) as origin.
    - Probes are uniformly distributed; spacing = minBrickSize / 3.
    - Convert each probe position to integer grid indices using the spacing.
    - Deduplicate probes mapped to the same grid index (keep the first).
    - Create dataset with rows: [x_norm, y_norm, z_norm, r, g, b] (float32).
    - Export EXR slices along Y axis: one EXR per y index, containing XZ plane RGB (L0).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading APV JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "minBrickSize" not in data:
        raise KeyError("JSON missing required key: minBrickSize")
    if "bricks" not in data:
        raise KeyError("JSON missing required key: bricks")

    min_brick_size = float(data["minBrickSize"])
    spacing = min_brick_size / 3.0
    if spacing <= 0:
        raise ValueError(f"Invalid spacing computed from minBrickSize: {min_brick_size}")

    # Gather probes
    pos_list = []
    rgb_list = []
    for brick in data["bricks"]:
        probes = brick.get("probes", [])
        for probe in probes:
            pos = probe.get("posWS", None)
            l0 = probe.get("L0", None)
            if pos is None or l0 is None:
                continue
            if len(pos) != 3 or len(l0) != 3:
                continue
            pos_list.append([float(pos[0]), float(pos[1]), float(pos[2])])
            rgb_list.append([float(l0[0]), float(l0[1]), float(l0[2])])

    if len(pos_list) == 0:
        raise ValueError("No probes found in JSON (expected bricks[*].probes[*].posWS and L0).")

    pos_np = np.asarray(pos_list, dtype=np.float64)  # (N, 3) in XYZ order
    rgb_np = np.asarray(rgb_list, dtype=np.float32)  # (N, 3) RGB (L0)

    origin = pos_np.min(axis=0)
    # Map positions to integer grid indices using known uniform spacing.
    # Use rint (bankers rounding) to be robust against tiny float errors; then shift to start from 0.
    idx_np = np.rint((pos_np - origin) / spacing).astype(np.int64)
    idx_np -= idx_np.min(axis=0)

    # Determine grid dimensions (Nx, Ny, Nz) in XYZ index space
    dims_xyz = (idx_np.max(axis=0) + 1).astype(np.int64)
    nx, ny, nz = int(dims_xyz[0]), int(dims_xyz[1]), int(dims_xyz[2])

    total_voxels = int(nx) * int(ny) * int(nz)
    print(f"Origin (min posWS): {origin.tolist()}")
    print(f"Spacing (minBrickSize/3): {spacing}")
    print(f"Grid dims (Nx, Ny, Nz): {(nx, ny, nz)} | total={total_voxels:,}")

    # Build dense volume indexed as [y, z, x] for convenient y-axis slicing.
    # Each slice is an XZ plane image.
    volume_yzx = np.zeros((ny, nz, nx, 3), dtype=np.float32)

    # Deduplicate by grid index (x,y,z)
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

    print(f"Probes: {idx_np.shape[0]} | unique grid points: {filled} | duplicates dropped: {dup_count}")

    # Optional sanity check: measure how well indices map back to positions
    recon_pos = origin + (idx_np.astype(np.float64) * spacing)
    max_abs_err = np.max(np.abs(recon_pos - pos_np))
    print(f"Max |posWS - (origin + idx*spacing)| = {max_abs_err:.6g}")

    # Build dataset (dense grid, same format as train_dataset_maker.py)
    dataset = np.zeros((total_voxels, 6), dtype=np.float32)
    out_min = None
    out_max = None
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
                    out_min = min(r, g, b)
                    out_max = max(r, g, b)
                else:
                    out_min = min(out_min, r, g, b)
                    out_max = max(out_max, r, g, b)
                k += 1
                

    print(f"NPY RGB range (after /scale_factor): min={out_min}, max={out_max}, scale_factor={scale_factor}")

    saved_path = os.path.join(output_dir, "train.npy")
    np.save(saved_path, dataset)
    print(f"Saved {saved_path}, shape={dataset.shape} (rows=[x,y,z,r,g,b])")

    if save_exr:
        tex_dir = os.path.join(output_dir, "textures")
        os.makedirs(tex_dir, exist_ok=True)

        num_slices = ny
        if max_exr_slices is not None:
            num_slices = min(num_slices, int(max_exr_slices))

        for iy in range(num_slices):
            # (Z, X, 3) where Z is image height, X is image width.
            slice_data = np.transpose(volume_yzx[iy], (1, 0, 2))  # (nx, nz, 3)
            filename = f"ambient_slice_{iy}.exr"
            path = os.path.join(tex_dir, filename)
            exr_util.write_exr(path, slice_data)
        print(f"Saved EXR slices along Y: {num_slices} files to {tex_dir}")


def _build_arg_parser(default_json: str, default_out: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build train.npy + y-sliced EXR textures from Unity APV probe JSON.")
    p.add_argument("--json", dest="json_path", type=str, default=default_json, help="Path to APV exported JSON")
    p.add_argument("--out", dest="output_dir", type=str, default=default_out, help="Output directory (will save train.npy and textures/)")
    p.add_argument("--scale-factor", dest="scale_factor", type=float, default=1.0, help="Divide RGB by this factor in train.npy")
    p.add_argument("--no-exr", dest="save_exr", action="store_false", help="Disable EXR export")
    p.add_argument("--max-exr-slices", dest="max_exr_slices", type=int, default=None, help="Limit EXR slices for debugging")
    p.set_defaults(save_exr=True)
    return p


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    default_json = os.path.join(current_dir, "data", "APV_Bricks_L0_SampleScene.json")
    default_out = os.path.join(current_dir, "data")

    parser = _build_arg_parser(default_json, default_out)
    args = parser.parse_args()

    make_apv_train_dataset(
        args.json_path,
        args.output_dir,
        scale_factor=args.scale_factor,
        save_exr=args.save_exr,
        max_exr_slices=args.max_exr_slices,
    )

