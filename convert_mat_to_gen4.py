from pathlib import Path
import numpy as np
from scipy.io import loadmat

# setting both sensors to same size
ATIS_SIZE  = (240, 304)  # (H, W)
DAVIS_SIZE = (240, 304)  # (180, 240)  # (H, W) -> if your DAVIS is 240x180 (W x H)


def load_mat_struct(s):
    """Convert MATLAB struct to Python dict."""
    if isinstance(s, np.ndarray) and s.size == 1:
        s = s.item()

    if hasattr(s, "_fieldnames"):
        return {name: np.squeeze(getattr(s, name)) for name in s._fieldnames}

    if hasattr(s, "dtype") and getattr(s.dtype, "names", None):
        return {name: np.squeeze(s[name]) for name in s.dtype.names}

    raise TypeError(f"Unsupported MATLAB struct type: {type(s)}")


def infer_hw_from_name(name: str) -> tuple[int, int]:
    n = name.lower()
    if "atis" in n:
        return ATIS_SIZE
    if "davis" in n:
        return DAVIS_SIZE
    return ATIS_SIZE


def convert_events_to_dat(data: dict, output_path: Path, mat_path: Path, H: int, W: int):
    print(f"\nConverting events: {mat_path} -> {output_path}")
    td = load_mat_struct(data["TD"])

    # Read raw
    x  = np.asarray(td["x"]).astype(np.int32, copy=False)
    y  = np.asarray(td["y"]).astype(np.int32, copy=False)
    ts = np.asarray(td["ts"]).astype(np.uint32, copy=False)
    p  = np.asarray(td["p"]).astype(np.int16, copy=False)

    # Sort by timestamp
    order = np.argsort(ts, kind="mergesort")
    x, y, ts, p = x[order], y[order], ts[order], p[order]

    # Filter to sensor bounds (drop out-of-range)
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    dropped = int(np.size(in_bounds) - np.count_nonzero(in_bounds))
    if dropped > 0:
        x, y, ts, p = x[in_bounds], y[in_bounds], ts[in_bounds], p[in_bounds]

    # Pack to Prophesee-like word: 14 bits x, 14 bits y, 1 bit polarity
    x_u = x.astype(np.uint32)
    y_u = y.astype(np.uint32)
    p_bit = (p > 0).astype(np.uint32)

    data_word = (x_u & 0x3FFF) | ((y_u & 0x3FFF) << 14) | (p_bit << 28)

    out = np.empty(ts.size * 2, dtype=np.uint32)
    out[0::2] = ts
    out[1::2] = data_word
    out.tofile(output_path)

    print(f"  Sensor: H={H}, W={W}")
    print(f"  Dropped out-of-bounds events: {dropped:,}")
    print(f"  Wrote {ts.size:,} events to {output_path}")


def convert_labels_to_npy(
    data: dict,
    output_path: Path,
    mat_path: Path,
    H: int,
    W: int,
    bbox_size: float = 22.0,
    class_id: int = 0,
):
    print(f"\nConverting labels: {mat_path} -> {output_path}")

    if "Obj" not in data:
        print("  Warning: No 'Obj' field found in .mat file. Skipping label conversion.")
        return

    obj = load_mat_struct(data["Obj"])
    obj_x  = np.asarray(obj.get("x", []), dtype=np.float32)
    obj_y  = np.asarray(obj.get("y", []), dtype=np.float32)
    obj_id = np.asarray(obj.get("id", []), dtype=np.int32)
    obj_ts = np.asarray(obj.get("ts", []), dtype=np.uint64)

    if obj_ts.size == 0:
        print("  Warning: No object annotations found.")
        return

    if not (obj_x.size == obj_y.size == obj_id.size == obj_ts.size):
        raise ValueError(f"Obj fields length mismatch in {mat_path.name}")

    # Build boxes around (x,y) center
    half = bbox_size * 0.5
    x0 = obj_x - half
    y0 = obj_y - half
    x1 = obj_x + half
    y1 = obj_y + half

    # Crop to FOV
    x0c = np.clip(x0, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    x1c = np.clip(x1, 0, W - 1)
    y1c = np.clip(y1, 0, H - 1)

    wc = x1c - x0c
    hc = y1c - y0c

    # Remove degenerate boxes after cropping
    keep = (wc > 0) & (hc > 0)
    dropped = int(obj_ts.size - np.count_nonzero(keep))

    x0c, y0c, wc, hc = x0c[keep], y0c[keep], wc[keep], hc[keep]
    ts, tid = obj_ts[keep], obj_id[keep]

    label_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("ts", "<u8"),
        ("class_id", "u1"),
        ("track_id", "<i4"),
    ])

    labels = np.empty(ts.size, dtype=label_dtype)
    labels["x"] = x0c.astype(np.float32)
    labels["y"] = y0c.astype(np.float32)
    labels["w"] = wc.astype(np.float32)
    labels["h"] = hc.astype(np.float32)
    labels["ts"] = ts
    labels["class_id"] = np.uint8(class_id)
    labels["track_id"] = tid

    labels.sort(order="ts")
    np.save(output_path, labels)

    print(f"  Sensor: H={H}, W={W}")
    print(f"  Dropped boxes after crop/degenerate: {dropped:,}")
    print(f"  Wrote {labels.size:,} labels to {output_path}")
    print(f"  Unique tracks: {np.unique(labels['track_id']).size}")


def main():
    data_dir = Path("data/EBSSA")
    output_dir = Path("data/ebssa_gen4")
    output_dir.mkdir(parents=True, exist_ok=True)

    for mat_path in sorted(data_dir.glob("*.mat")):
        mat_name = mat_path.stem.replace("_td_labelled", "")
        H, W = infer_hw_from_name(mat_name)

        output_dat = output_dir / f"{mat_name}_td.dat"
        output_npy = output_dir / f"{mat_name}_bbox.npy"

        data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

        convert_events_to_dat(data, output_dat, mat_path, H=H, W=W)
        convert_labels_to_npy(data, output_npy, mat_path, H=H, W=W, bbox_size=23, class_id=0)

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
