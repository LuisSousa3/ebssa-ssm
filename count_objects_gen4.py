#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import os

def get_dat_duration(dat_path: Path) -> float:
    """
    Reads the last 4-byte timestamp from the .dat file without loading the whole file.
    Assumes 8-byte event records (4 bytes TS, 4 bytes Data).
    Returns duration in seconds.
    """
    if not dat_path.exists() or dat_path.stat().st_size < 8:
        return 0.0
    
    with open(dat_path, "rb") as f:
        # Seek to 8 bytes before the end of the file
        f.seek(-8, os.SEEK_END)
        # Read the first 4 bytes of that last record (the timestamp)
        last_ts = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    
    # Timestamps are usually in microseconds (μs)
    return float(last_ts) / 1e6

def count_objects_from_labels(labels: np.ndarray) -> int:
    """Count unique tracked objects using obj_id field."""
    if labels.size == 0:
        return 0
    return int(np.unique(labels["track_id"]).size)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/ebssa_gen4") # Updated to match your conversion output
    args = ap.parse_args()

    folder = Path(args.folder)
    bbox_paths = sorted(folder.glob("*_bbox.npy"))
    
    if not bbox_paths:
        raise SystemExit(f"No *_bbox.npy found in {folder}")

    rows = []
    
    print(f"{'Recording':<25} | {'Objects':<8} | {'Duration (s)':<12}")
    print("-" * 50)

    for bbox_path in bbox_paths:
        stem = bbox_path.name.replace("_bbox.npy", "")
        td_path = folder / f"{stem}_td.dat"

        if not td_path.exists():
            continue

        try:
            labels = np.load(bbox_path)
            nobj = count_objects_from_labels(labels)
            duration = get_dat_duration(td_path)
            
            rows.append({
                "recording": stem,
                "num_objects": nobj,
                "duration": duration
            })
            
            print(f"{stem:<25} | {nobj:<8} | {duration:<12.2f}")

        except Exception as e:
            print(f"[WARN] Failed to process {stem}: {e}")

    if rows:
        total_objs = sum(r['num_objects'] for r in rows)
        avg_duration = sum(r['duration'] for r in rows) / len(rows)
        print("-" * 50)
        print(f"Total Unique Objects: {total_objs}")
        print(f"Average Duration:     {avg_duration:.2f}s")

if __name__ == "__main__":
    main()