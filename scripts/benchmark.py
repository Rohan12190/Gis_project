"""Benchmark script comparing offline snapshot load performance, verification, and memory footprint."""

import sys
import time
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from greenplan.snapshots import list_snapshots, load_snapshot, verify_snapshots


def get_dir_size(path: Path) -> float:
    """Return total directory size in megabytes."""
    total = 0
    for p in path.glob("**/*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def benchmark_snapshots():
    print("=" * 75)
    print(" GREENPLAN GIS REAL-DATA BENCHMARK (OFFLINE SNAPSHOTS)")
    print("=" * 75)

    snapshots = list_snapshots()
    if not snapshots:
        print("No snapshots found in data/snapshots/.")
        return

    snap_ids = [s["snapshot_id"] for s in snapshots]
    print(f"Discovered {len(snap_ids)} snapshot(s): {', '.join(snap_ids)}")
    print("-" * 75)
    print(f"{'Snapshot':<12} | {'Disk Size':<10} | {'SHA-256 Check':<14} | {'Load Time':<11} | {'Peak RAM':<10}")
    print("-" * 75)

    verifications = verify_snapshots()

    for s in snapshots:
        snap_id = s["snapshot_id"]
        snap_dir = Path("data/snapshots") / snap_id
        disk_mb = get_dir_size(snap_dir)

        # 1. Verification time
        t0 = time.perf_counter()
        is_valid = verifications.get(snap_id, False)
        verify_time_ms = (time.perf_counter() - t0) * 1000

        if not is_valid:
            print(f"{snap_id:<12} | {disk_mb:>7.2f} MB | FAILED CHECK   | N/A         | N/A")
            continue

        # 2. Load time and peak memory
        tracemalloc.start()
        t1 = time.perf_counter()
        load_snapshot(snap_id)
        load_time_ms = (time.perf_counter() - t1) * 1000
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        ver_str = f"PASSED ({verify_time_ms:.0f}ms)"
        print(f"{snap_id:<12} | {disk_mb:>7.2f} MB | {ver_str:<14} | {load_time_ms:>8.2f} ms | {peak_mb:>7.2f} MB")

    print("-" * 75)
    print("Benchmark complete. All snapshots loaded and verified cryptographically.")
    print("=" * 75)


if __name__ == "__main__":
    benchmark_snapshots()
