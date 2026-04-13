#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def stem_from_suffix(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        return ""
    return name[: -len(suffix)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ebssa_gen4", help="source folder")
    ap.add_argument("--dst", default="data/gen4_full_video", help="destination folder")
    ap.add_argument("--dry-run", action="store_true", help="do not copy, only print")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    td_files = {stem_from_suffix(p.name, "_td.dat"): p for p in src.glob("*_td.dat")}
    bb_files = {stem_from_suffix(p.name, "_bbox.npy"): p for p in src.glob("*_bbox.npy")}

    td_stems = {k for k in td_files.keys() if k}
    bb_stems = {k for k in bb_files.keys() if k}

    paired = sorted(td_stems & bb_stems)
    only_td = sorted(td_stems - bb_stems)
    only_bb = sorted(bb_stems - td_stems)

    print(f"Source: {src}")
    print(f"Found *_td.dat:   {len(td_stems)}")
    print(f"Found *_bbox.npy: {len(bb_stems)}")
    print(f"Paired:           {len(paired)}")
    print(f"Only td:          {len(only_td)}")
    print(f"Only bbox:        {len(only_bb)}")

    if args.dry_run:
        print("\nDry run: not copying. First 10 paired stems:")
        for s in paired[:10]:
            print("  ", s)
        return

    copied = 0
    for s in paired:
        for p in (td_files[s], bb_files[s]):
            out = dst / p.name
            shutil.copy2(p, out)
        copied += 1

    print(f"\nCopied {copied} paired recordings (2 files each) into: {dst}")


if __name__ == "__main__":
    main()