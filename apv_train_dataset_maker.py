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


def _make_subcell_offsets(subdiv: int, mode: str) -> np.ndarray:
    """
    Returns offsets in (approximately) [0,1] for sampling *inside* a unit cell.

    - mode="cell_center": (k+0.5)/subdiv, avoids duplicating cell corners
    - mode="include_corners": linspace(0,1,subdiv), includes corners (may duplicate original grid samples)
    """
    if subdiv <= 0:
        raise ValueError(f"subdiv must be > 0, got {subdiv}")
    if mode == "cell_center":
        return (np.arange(subdiv, dtype=np.float32) + 0.5) / float(subdiv)
    if mode == "include_corners":
        if subdiv == 1:
            return np.asarray([0.0], dtype=np.float32)
        return np.linspace(0.0, 1.0, num=subdiv, endpoint=True, dtype=np.float32)
    raise ValueError(f"Unknown densify mode: {mode} (expected 'cell_center' or 'include_corners')")


def densify_volume_trilinear_to_dataset(
    volume_yzx: np.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    subdiv: int = 4,
    scale_factor: float = 1.0,
    mode: str = "cell_center",
    chunk_cells: int = 8192,
) -> np.ndarray:
    """
    Densify a regular grid volume into additional training samples.

    For each unit cell formed by 8 corner grid points:
      (ix,iy,iz) .. (ix+1,iy+1,iz+1),
    generate subdiv^3 sample points uniformly distributed in the cell (default: 4^3=64).
    RGB values are computed by trilinear interpolation of the 8 corners.

    Returns float32 array of shape (M, 6): [x_norm, y_norm, z_norm, r, g, b]
    where x_norm/y_norm/z_norm are in [0,1] following the same normalization as _safe_norm.
    """
    if volume_yzx.ndim != 4 or volume_yzx.shape[-1] != 3:
        raise ValueError(f"volume_yzx must have shape (ny,nz,nx,3), got {volume_yzx.shape}")
    if nx <= 1 or ny <= 1 or nz <= 1:
        # No cells exist along at least one axis
        return np.zeros((0, 6), dtype=np.float32)
    if subdiv <= 0:
        raise ValueError(f"subdiv must be > 0, got {subdiv}")
    if chunk_cells <= 0:
        chunk_cells = 8192

    cells_x = int(nx - 1)
    cells_y = int(ny - 1)
    cells_z = int(nz - 1)
    num_cells = int(cells_x) * int(cells_y) * int(cells_z)

    t = _make_subcell_offsets(subdiv, mode)
    uu, vv, ww = np.meshgrid(t, t, t, indexing="ij")
    u = uu.reshape(-1).astype(np.float32, copy=False)  # (S,)
    v = vv.reshape(-1).astype(np.float32, copy=False)  # (S,)
    w = ww.reshape(-1).astype(np.float32, copy=False)  # (S,)
    s = int(u.shape[0])  # S = subdiv^3

    # Trilinear weights for 8 corners, order:
    # 000, 100, 010, 110, 001, 101, 011, 111
    w000 = (1.0 - u) * (1.0 - v) * (1.0 - w)
    w100 = u * (1.0 - v) * (1.0 - w)
    w010 = (1.0 - u) * v * (1.0 - w)
    w110 = u * v * (1.0 - w)
    w001 = (1.0 - u) * (1.0 - v) * w
    w101 = u * (1.0 - v) * w
    w011 = (1.0 - u) * v * w
    w111 = u * v * w
    weights = np.stack([w000, w100, w010, w110, w001, w101, w011, w111], axis=1).astype(np.float32, copy=False)  # (S,8)

    denom_x = np.float32(nx - 1)
    denom_y = np.float32(ny - 1)
    denom_z = np.float32(nz - 1)

    out = np.empty((num_cells * s, 6), dtype=np.float32)
    write = 0

    for start in range(0, num_cells, chunk_cells):
        end = min(start + chunk_cells, num_cells)
        ids = np.arange(start, end, dtype=np.int64)

        # Flattening order: ix fastest, then iz, then iy
        ix = ids % cells_x
        tmp = ids // cells_x
        iz = tmp % cells_z
        iy = tmp // cells_z

        # Gather 8 corners (B,3)
        c000 = volume_yzx[iy, iz, ix]
        c100 = volume_yzx[iy, iz, ix + 1]
        c010 = volume_yzx[iy + 1, iz, ix]
        c110 = volume_yzx[iy + 1, iz, ix + 1]
        c001 = volume_yzx[iy, iz + 1, ix]
        c101 = volume_yzx[iy, iz + 1, ix + 1]
        c011 = volume_yzx[iy + 1, iz + 1, ix]
        c111 = volume_yzx[iy + 1, iz + 1, ix + 1]
        corners = np.stack([c000, c100, c010, c110, c001, c101, c011, c111], axis=1)  # (B,8,3)

        # Interpolate RGB: (B,S,3)
        rgb = np.einsum("sa,bac->bsc", weights, corners).astype(np.float32, copy=False)
        if float(scale_factor) != 1.0:
            rgb = rgb / np.float32(scale_factor)

        # Coordinates: (B,S)
        ix_f = ix.astype(np.float32, copy=False)
        iy_f = iy.astype(np.float32, copy=False)
        iz_f = iz.astype(np.float32, copy=False)
        x_norm = (ix_f[:, None] + u[None, :]) / denom_x
        y_norm = (iy_f[:, None] + v[None, :]) / denom_y
        z_norm = (iz_f[:, None] + w[None, :]) / denom_z

        bs = int((end - start) * s)
        out_slice = out[write: write + bs]
        out_slice[:, 0] = x_norm.reshape(-1)
        out_slice[:, 1] = y_norm.reshape(-1)
        out_slice[:, 2] = z_norm.reshape(-1)
        out_slice[:, 3:] = rgb.reshape(-1, 3)
        write += bs

    return out


def make_apv_train_dataset(
    json_path: str,
    output_dir: str,
    *,
    scale_factor: float = 1.0,
    densify: bool = False,
    densify_subdiv: int = 4,
    densify_mode: str = "cell_center",
    densify_chunk_cells: int = 8192,
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
    maxPos = pos_np.max(axis=0)
    # Map positions to integer grid indices using known uniform spacing.
    # Use rint (bankers rounding) to be robust against tiny float errors; then shift to start from 0.
    idx_np = np.rint((pos_np - origin) / spacing).astype(np.int64)
    idx_np -= idx_np.min(axis=0)

    # Determine grid dimensions (Nx, Ny, Nz) in XYZ index space
    dims_xyz = (idx_np.max(axis=0) + 1).astype(np.int64)
    nx, ny, nz = int(dims_xyz[0]), int(dims_xyz[1]), int(dims_xyz[2])

    total_voxels = int(nx) * int(ny) * int(nz)
    print(f"Origin (min posWS): {origin.tolist()}")
    print(f"Max posWS: {maxPos.tolist()}")
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

    # Always save a non-densified reference set for evaluation (original ix/iy/iz grid)
    eval_path = os.path.join(output_dir, "eval.npy")
    np.save(eval_path, dataset)
    print(f"Saved {eval_path}, shape={dataset.shape} (base grid, rows=[x,y,z,r,g,b])")

    train_dataset = dataset
    if densify:
        if densify_subdiv <= 0:
            raise ValueError(f"densify_subdiv must be > 0, got {densify_subdiv}")

        num_cells = max(nx - 1, 0) * max(ny - 1, 0) * max(nz - 1, 0)
        dense_points_per_cell = int(densify_subdiv) ** 3
        dense_total = int(num_cells) * int(dense_points_per_cell)
        print(
            f"Densify enabled: cells={(nx-1, ny-1, nz-1)} (total={num_cells:,}), "
            f"subdiv={densify_subdiv} -> {dense_points_per_cell} samples/cell, "
            f"dense_samples={dense_total:,}"
        )

        dense_dataset = densify_volume_trilinear_to_dataset(
            volume_yzx,
            nx=nx,
            ny=ny,
            nz=nz,
            subdiv=int(densify_subdiv),
            scale_factor=float(scale_factor),
            mode=str(densify_mode),
            chunk_cells=int(densify_chunk_cells),
        )
        print(f"Dense dataset built: shape={dense_dataset.shape}")

        train_dataset = np.concatenate([dataset, dense_dataset], axis=0)
        del dense_dataset
        rgb_min = float(train_dataset[:, 3:].min()) if train_dataset.size else 0.0
        rgb_max = float(train_dataset[:, 3:].max()) if train_dataset.size else 0.0
        print(f"Train NPY RGB range (after /scale_factor): min={rgb_min}, max={rgb_max}")

    saved_path = os.path.join(output_dir, "train.npy")
    np.save(saved_path, train_dataset)
    print(f"Saved {saved_path}, shape={train_dataset.shape} (rows=[x,y,z,r,g,b])")

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
    p.add_argument(
        "--densify",
        dest="densify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Densify training samples (default: True). Use --no-densify to disable. Also saves eval.npy as base grid.",
    )
    p.add_argument(
        "--densify-subdiv",
        dest="densify_subdiv",
        type=int,
        default=2,
        help="Samples per axis inside each cell (default 4 => 64 samples/cell).",
    )
    p.add_argument(
        "--densify-mode",
        dest="densify_mode",
        type=str,
        default="cell_center",
        choices=["cell_center", "include_corners"],
        help="Sampling positions inside each cell. 'cell_center' avoids duplicating original grid points.",
    )
    p.add_argument(
        "--densify-chunk-cells",
        dest="densify_chunk_cells",
        type=int,
        default=8192,
        help="How many cells to process per chunk (controls peak memory).",
    )
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
        densify=args.densify,
        densify_subdiv=args.densify_subdiv,
        densify_mode=args.densify_mode,
        densify_chunk_cells=args.densify_chunk_cells,
        save_exr=args.save_exr,
        max_exr_slices=args.max_exr_slices,
    )

