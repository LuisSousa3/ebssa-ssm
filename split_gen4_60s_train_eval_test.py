#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_group_and_segment(stem: str) -> Tuple[str, str]:
    """
    Given a filename stem like:
      archenar_leos_11_33_atis_0001
    Return:
      group = archenar_leos_11_33_atis
      seg   = 0001

    Assumes the last underscore-separated token is the segment index.
    """
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected stem format: {stem}")
    seg = parts[-1]
    group = "_".join(parts[:-1])
    return group, seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-folder", default="data/gen4_60s", help="folder with 60s segmented files")
    ap.add_argument("--out-folder", default="data/gen4", help="output folder with train/val/test subfolders")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen, do not copy/move")
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("train+val+test must sum to 1.0")

    in_dir = Path(args.in_folder)
    out_dir = Path(args.out_folder)

    # Output subfolders
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    td_files = sorted(in_dir.glob("*_td.dat"))
    if not td_files:
        raise SystemExit(f"No *_td.dat found in {in_dir}")

    # Build list of paired segment stems
    paired_stems: List[str] = []
    for td in td_files:
        stem = td.name[:-len("_td.dat")]
        bbox = in_dir / f"{stem}_bbox.npy"
        if bbox.exists():
            paired_stems.append(stem)

    if not paired_stems:
        raise SystemExit(f"No paired *_td.dat + *_bbox.npy found in {in_dir}")

    # Group stems by recording prefix
    groups: Dict[str, List[str]] = {}
    for stem in paired_stems:
        group, _seg = parse_group_and_segment(stem)
        groups.setdefault(group, []).append(stem)

    # Shuffle groups deterministically
    group_names = sorted(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_names)

    n_groups = len(group_names)
    n_train = int(round(n_groups * args.train))
    n_val = int(round(n_groups * args.val))
    # ensure total sums correctly
    n_test = n_groups - n_train - n_val

    train_groups = set(group_names[:n_train])
    val_groups = set(group_names[n_train:n_train + n_val])
    test_groups = set(group_names[n_train + n_val:])

    def split_name(group: str) -> str:
        if group in train_groups:
            return "train"
        if group in val_groups:
            return "val"
        return "test"

    # Copy/move files
    op = shutil.move if args.move else shutil.copy2

    counts = {"train": 0, "val": 0, "test": 0}
    seg_counts = {"train": 0, "val": 0, "test": 0}

    for group, stems in groups.items():
        split = split_name(group)
        target_dir = {"train": train_dir, "val": val_dir, "test": test_dir}[split]

        for stem in stems:
            td_src = in_dir / f"{stem}_td.dat"
            bb_src = in_dir / f"{stem}_bbox.npy"
            td_dst = target_dir / td_src.name
            bb_dst = target_dir / bb_src.name

            if args.dry_run:
                continue

            op(td_src, td_dst)
            op(bb_src, bb_dst)

            counts[split] += 2
            seg_counts[split] += 1

    print(f"Input folder: {in_dir}")
    print(f"Found paired segments: {len(paired_stems)}")
    print(f"Unique recordings (groups): {n_groups}")
    print(f"Split by recording groups with seed={args.seed}:")
    print(f"  train groups: {len(train_groups)}")
    print(f"  val  groups: {len(val_groups)}")
    print(f"  test  groups: {len(test_groups)}")

    print("\nFiles copied/moved (2 per segment):")
    print(f"  train: {seg_counts['train']} segments -> {counts['train']} files")
    print(f"  val : {seg_counts['val']} segments -> {counts['val']} files")
    print(f"  test : {seg_counts['test']} segments -> {counts['test']} files")

    if args.dry_run:
        print("\n(dry-run) No files were copied/moved.")
    else:
        print(f"\nOutput folders:")
        print(f"  {train_dir}")
        print(f"  {val_dir}")
        print(f"  {test_dir}")


if __name__ == "__main__":
    main()
