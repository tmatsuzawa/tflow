#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py


def _read_pivlab_txt(fp: Path) -> tuple[list[str], np.ndarray]:
    # PIVlab ASCII chart output:
    # line 1: text
    # line 2: text (FRAME: ...)
    # line 3: CSV column names
    # line 4+: numeric CSV rows
    with fp.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        _ = f.readline()
        col_line = f.readline().strip()

    cols = [c.strip() for c in col_line.split(",")]
    data = np.loadtxt(fp, delimiter=",", skiprows=3)

    if data.ndim != 2:
        raise ValueError(f"{fp.name}: expected 2D table, got {data.shape}")
    if data.shape[1] != len(cols):
        raise ValueError(f"{fp.name}: header has {len(cols)} cols but data has {data.shape[1]}")
    return cols, data


def _find_col(cols: list[str], needle: str) -> int | None:
    needle_l = needle.lower()
    for i, c in enumerate(cols):
        if needle_l in c.lower():
            return i
    return None


def _build_grid_mapping(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    N = x_unique.size
    M = y_unique.size
    if M * N != x.size:
        raise ValueError(f"Not a rectangular grid: {x.size} points vs {M}*{N}")

    x2d, y2d = np.meshgrid(x_unique, y_unique)  # (M, N)

    x_index = {val: j for j, val in enumerate(x_unique)}
    y_index = {val: i for i, val in enumerate(y_unique)}
    cols = np.fromiter((x_index[val] for val in x), dtype=np.int64, count=x.size)
    rows = np.fromiter((y_index[val] for val in y), dtype=np.int64, count=y.size)
    return x2d, y2d, rows, cols


def pivlab2hdf5(
    input_dir: Path,
    *,
    scale: float = 1.0,          # um/px
    dt: float | None = None,     # s/frame
    fps: float | None = None,    # frames/s (used only if dt is None)
    overwrite: bool = False,
    compression: str | None = "gzip",
) -> Path:
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")

    output_h5 = input_dir.parent / f"{input_dir.name}.h5"

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {input_dir}")

    if output_h5.exists():
        if overwrite:
            output_h5.unlink()
        else:
            raise FileExistsError(f"{output_h5} exists (use --overwrite)")

    if scale <= 0:
        raise ValueError("scale must be > 0 (um/px)")

    # Resolve dt using dt/fps rules
    dt_source = "dt"
    if dt is None:
        if fps is not None:
            if fps <= 0:
                raise ValueError("fps must be > 0")
            dt = 1.0 / fps
            dt_source = "fps"
        else:
            dt = 1.0
            dt_source = "default(1.0)"

    if dt <= 0:
        raise ValueError("dt must be > 0 (s/frame)")

    # First file defines columns + grid mapping
    cols0, d0 = _read_pivlab_txt(files[0])

    ix = _find_col(cols0, "x [")
    iy = _find_col(cols0, "y [")
    iu = _find_col(cols0, "u [")
    iv = _find_col(cols0, "v [")
    iomega = _find_col(cols0, "vorticity")

    if None in (ix, iy, iu, iv):
        raise KeyError(f"Missing required columns x,y,u,v in header: {cols0}")

    x_px = d0[:, ix]
    y_px = d0[:, iy]
    x2d_px, y2d_px, rows, cols = _build_grid_mapping(x_px, y_px)

    # Convert x,y to um
    x2d = x2d_px.astype(np.float32) * float(scale)
    y2d = y2d_px.astype(np.float32) * float(scale)

    M, N = x2d.shape
    T = len(files)

    # Time array (seconds)
    t_arr = (np.arange(T, dtype=np.float32) * float(dt))

    # Convert velocities px/frame -> um/s
    vel_factor = float(scale) / float(dt)

    ux = np.empty((M, N, T), dtype=np.float32)
    uy = np.empty((M, N, T), dtype=np.float32)

    omega = np.empty((M, N, T), dtype=np.float32) if iomega is not None else None
    omega_factor = 1.0 / float(dt)  # assumes omega is in 1/frame

    # Fill all frames
    for ti, fp in enumerate(files):
        cols_t, d = _read_pivlab_txt(fp)
        if cols_t != cols0:
            raise ValueError(f"{fp.name}: column header differs from first file")
        if d.shape[0] != d0.shape[0]:
            raise ValueError(f"{fp.name}: point count differs from first file")

        ux[rows, cols, ti] = d[:, iu].astype(np.float32, copy=False) * vel_factor
        uy[rows, cols, ti] = d[:, iv].astype(np.float32, copy=False) * vel_factor
        if omega is not None:
            omega[rows, cols, ti] = d[:, iomega].astype(np.float32, copy=False) * omega_factor

    # Chunking: one time-slice per chunk
    chunks = (M, N, 1)

    with h5py.File(output_h5, "w") as f:
        # Attributes (always present)
        f.attrs["scale"] = float(scale)   # um/px
        f.attrs["dt"] = float(dt)         # s/frame (resolved)
        f.attrs["fps"] = float(fps) if fps is not None else np.nan
        f.attrs["dt_source"] = dt_source

        # Required datasets
        f.create_dataset("x", data=x2d, compression=compression)
        f.create_dataset("y", data=y2d, compression=compression)
        f.create_dataset("t", data=t_arr, compression=compression)

        f.create_dataset("ux", data=ux, compression=compression, chunks=chunks)
        f.create_dataset("uy", data=uy, compression=compression, chunks=chunks)
        if omega is not None:
            f.create_dataset("omega", data=omega, compression=compression, chunks=chunks)

    return output_h5


def main() -> None:
    p = argparse.ArgumentParser(description="Convert folder of PIVlab txt files into one HDF5.")
    p.add_argument("input_dir", type=Path, help="Folder containing *.txt")
    p.add_argument("-scale", type=float, default=1.0, help="Spatial scale (um/px)")
    p.add_argument("-dt", type=float, default=None, help="Time step (s/frame). Overrides fps if both given.")
    p.add_argument("-fps", type=float, default=None, help="Frames per second. Used only if dt is not given.")
    p.add_argument("-overwrite", action="store_true", help="Overwrite existing output .h5")
    args = p.parse_args()

    out = pivlab2hdf5(
        args.input_dir,
        scale=args.scale,
        dt=args.dt,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
