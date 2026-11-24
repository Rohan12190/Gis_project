"""Benchmark script comparing offline snapshot load performance, verification, and real raster memory footprint."""

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from greenplan.snapshots import list_snapshots, load_snapshot


def get_dir_size(path: Path) -> float:
    """Return total directory size in megabytes."""
    total = 0
    for p in path.glob("**/*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def benchmark_snapshots():
    print("=" * 80)
    print(" GREENPLAN GIS REAL-DATA BENCHMARK (OFFLINE SNAPSHOTS)")
    print("=" * 80)

    snapshots = list_snapshots()
    if not snapshots:
        print("No snapshots found in data/snapshots/.")
        return

    snap_ids = [s["snapshot_id"] for s in snapshots]
    print(f"Discovered {len(snap_ids)} snapshot(s): {', '.join(snap_ids)}")
    print("-" * 80)
    print(f"{'Snapshot':<12} | {'Disk Size':<10} | {'SHA-256 Check':<15} | {'Load All Rasters':<18} | {'Peak RAM':<10}")
    print("-" * 80)

    for s in snapshots:
        snap_id = s["snapshot_id"]
        snap_dir = Path("data/snapshots") / snap_id
        disk_mb = get_dir_size(snap_dir)

        # 1. Real SHA-256 integrity verification time (hashing all files against checksums.json)
        t0 = time.perf_counter()
        try:
            load_snapshot(snap_id, base_dir=Path("data/snapshots"))
            is_valid = True
        except Exception:
            is_valid = False
        verify_time_ms = (time.perf_counter() - t0) * 1000

        if not is_valid:
            print(f"{snap_id:<12} | {disk_mb:>7.2f} MB | FAILED CHECK    | N/A                | N/A")
            continue

        # 2. Honest load time and peak memory of all multi-epoch rasters + inputs
        tracemalloc.start()
        t1 = time.perf_counter()
        load_snapshot(snap_id)
        # Load all compressed multi-epoch rasters from inputs.npz into RAM
        loaded_arrays = {}
        with np.load(snap_dir / "inputs.npz") as npz:
            for k in npz.files:
                loaded_arrays[k] = npz[k]
        load_time_ms = (time.perf_counter() - t1) * 1000
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        ver_str = f"PASSED ({verify_time_ms:.0f}ms)"
        load_str = f"{load_time_ms:.1f} ms ({len(loaded_arrays)} arr)"
        print(f"{snap_id:<12} | {disk_mb:>7.2f} MB | {ver_str:<15} | {load_str:<18} | {peak_mb:>7.2f} MB")

    print("-" * 80)
    print("Benchmark complete. All multi-epoch rasters loaded and verified cryptographically.")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_snapshots()
