#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def split_events_dat(
    in_dat: Path,
    out_dir: Path,
    base_stem: str,
    window_us: int,
    shift_ts: bool,
    block_events: int = 5_000_000,
) -> int:
    """
    Split GEN4-style .dat file where content is uint32 pairs:
      [ts0, data0, ts1, data1, ...]
    Writes segments: {base_stem}_{idx:04d}_td.dat
    Returns number of segments written.
    """
    raw = np.memmap(in_dat, dtype=np.uint32, mode="r")
    if raw.size % 2 != 0:
        raise ValueError(f"{in_dat} size not divisible by 8 bytes (uint32 pairs).")

    n_events = raw.size // 2
    if n_events == 0:
        return 0

    # First timestamp (absolute)
    t0 = int(raw[0])

    seg_files: dict[int, object] = {}
    seg_written = set()

    def get_fp(seg_idx: int):
        if seg_idx not in seg_files:
            out_path = out_dir / f"{base_stem}_{seg_idx:03d}_td.dat"
            seg_files[seg_idx] = open(out_path, "wb")
            seg_written.add(seg_idx)
        return seg_files[seg_idx]

    # Stream through file in chunks of events
    start = 0
    while start < n_events:
        end = min(n_events, start + block_events)
        # slice raw as pairs
        chunk = np.asarray(raw[start * 2 : end * 2], dtype=np.uint32).reshape(-1, 2)
        ts = chunk[:, 0].astype(np.int64)
        data = chunk[:, 1]

        seg_idx = ((ts - t0) // window_us).astype(np.int64) + 1  # 1-based
        # Write each segment portion
        for s in np.unique(seg_idx):
            mask = seg_idx == s
            ts_s = ts[mask]
            data_s = data[mask]

            if shift_ts:
                seg_start = t0 + (int(s) - 1) * window_us
                ts_s = (ts_s - seg_start).astype(np.uint32)
            else:
                ts_s = ts_s.astype(np.uint32)

            out = np.empty(ts_s.size * 2, dtype=np.uint32)
            out[0::2] = ts_s
            out[1::2] = data_s.astype(np.uint32)

            fp = get_fp(int(s))
            out.tofile(fp)

        start = end

    # Close files
    for fp in seg_files.values():
        fp.close()

    return len(seg_written)


def split_labels_npy(
    in_npy: Path,
    out_dir: Path,
    base_stem: str,
    window_us: int,
    shift_ts: bool,
    t0: int,
) -> int:
    """
    Split structured labels array by ts into segment files:
      {base_stem}_{idx:04d}_bbox.npy
    Returns number of segment label files written.
    """
    labels = np.load(in_npy)
    if labels.size == 0:
        return 0

    if labels.dtype.names is None or "ts" not in labels.dtype.names:
        raise ValueError(f"{in_npy} does not look like the expected structured label array (missing 'ts').")

    ts = labels["ts"].astype(np.int64)
    seg_idx = ((ts - t0) // window_us).astype(np.int64) + 1

    written = 0
    for s in np.unique(seg_idx):
        mask = seg_idx == s
        seg_labels = labels[mask].copy()

        if shift_ts:
            seg_start = t0 + (int(s) - 1) * window_us
            seg_labels["ts"] = (seg_labels["ts"].astype(np.int64) - seg_start).astype(np.uint64)

        out_path = out_dir / f"{base_stem}_{int(s):03d}_bbox.npy"
        np.save(out_path, seg_labels)
        written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-folder", default="data/gen4_full_video", help="folder containing paired *_td.dat and *_bbox.npy")
    ap.add_argument("--out-folder", default="data/gen4_60s", help="output folder for segmented files")
    ap.add_argument("--window-sec", type=int, default=60, help="segment length in seconds")
    ap.add_argument("--shift-ts", action="store_true", help="shift timestamps so each segment starts at ts=0")
    ap.add_argument("--no-shift-ts", dest="shift_ts", action="store_false", help="keep absolute timestamps")
    ap.set_defaults(shift_ts=True)
    args = ap.parse_args()

    in_folder = Path(args.in_folder)
    out_folder = Path(args.out_folder)
    ensure_dir(out_folder)

    window_us = int(args.window_sec) * 1_000_000

    dat_paths = sorted(in_folder.glob("*_td.dat"))
    if not dat_paths:
        raise SystemExit(f"No *_td.dat found in {in_folder}")

    total_segments = 0
    for dat_path in dat_paths:
        base_stem = dat_path.name[: -len("_td.dat")]
        npy_path = in_folder / f"{base_stem}_bbox.npy"
        if not npy_path.exists():
            # skip unpaired
            continue

        # Determine recording t0 from events (first uint32)
        raw0 = np.memmap(dat_path, dtype=np.uint32, mode="r")
        if raw0.size < 2:
            print(f"[WARN] Empty/invalid dat: {dat_path.name}")
            continue
        t0 = int(raw0[0])

        print(f"\n== {base_stem} ==")
        print(f"  t0 (from events): {t0}")

        nseg_events = split_events_dat(
            dat_path, out_folder, base_stem, window_us, args.shift_ts
        )
        nseg_labels = split_labels_npy(
            npy_path, out_folder, base_stem, window_us, args.shift_ts, t0
        )

        print(f"  segments written (events): {nseg_events}")
        print(f"  segments written (labels): {nseg_labels}")

        # Ideally should match, but if a segment has no labels it can differ.
        total_segments += max(nseg_events, nseg_labels)

    print(f"\nDone. Total segments created (approx): {total_segments}")
    print(f"Output folder: {out_folder}")


if __name__ == "__main__":
    main()
