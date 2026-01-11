# select_32_channels.py
import os, json, numpy as np
from pathlib import Path

# canonical 32-channel list (10-20-ish) — edit if you want another list
TARGET_CHANS = [
 "Fp1","Fp2","F7","F3","Fz","F4","F8",
 "FT9","FT10","T7","C3","Cz","C4","T8",
 "TP9","TP10","P7","P3","Pz","P4","P8",
 "PO9","O1","Oz","O2","PO10","AF3","AF4","FC1","FC2","CP1","CP2"
]

def normalize_name(s):
    return s.strip().upper().replace(" ", "").replace("-", "").replace(".", "")

def build_index_map(src_ch_names, target_list):
    src_norm = [normalize_name(c) for c in src_ch_names]
    idx_map = []
    for t in target_list:
        tn = normalize_name(t)
        if tn in src_norm:
            idx_map.append(src_norm.index(tn))
        else:
            idx_map.append(None)
    return idx_map

def process_folder(epoch_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    # find epoch npy files
    for p in sorted(Path(epoch_folder).glob("*_epoch*.npy")):
        base = p.stem
        meta_path = str(p).replace(".npy", "_epoch_summary.json")
        # fallback to any meta near file (older naming)
        if not os.path.exists(meta_path):
            meta_path = str(p).replace("_epoch","_meta")  # fallback attempt
            if meta_path.endswith(".json") and not os.path.exists(meta_path):
                meta_path = None

        # load epoch data
        try:
            ep = np.load(p)  # shape (channels, samples)
        except Exception as e:
            print("SKIP load failed:", p, e)
            continue

        # attempt to find source channel names in sibling meta files
        src_chs = None
        big_meta = None
        # try to find cleaned meta in same folder or parent
        candidate_meta = list(Path(p).parent.glob("*_meta.json")) + list(Path(p).parent.glob("*_cleaned_meta.json"))
        if meta_path and os.path.exists(meta_path):
            try:
                big_meta = json.load(open(meta_path, 'r'))
                src_chs = big_meta.get("ch_names") or big_meta.get("channel_names")
            except Exception:
                src_chs = None
        if not src_chs:
            for m in candidate_meta:
                try:
                    mm = json.load(open(m, 'r'))
                    if "ch_names" in mm:
                        src_chs = mm["ch_names"]
                        break
                except Exception:
                    continue

        if src_chs is None:
            # fallback: assume channels = range
            nchan = ep.shape[0]
            src_chs = [f"CH{i+1}" for i in range(nchan)]

        idx_map = build_index_map(src_chs, TARGET_CHANS)

        # build output array (32, samples) fill zeros for missing
        out_arr = np.zeros((len(TARGET_CHANS), ep.shape[1]), dtype=ep.dtype)
        for i, src_idx in enumerate(idx_map):
            if src_idx is None:
                # leave zeros
                continue
            if src_idx < ep.shape[0]:
                out_arr[i] = ep[src_idx]
            else:
                # unexpected, keep zeros
                pass

        out_base = base + "_32ch"
        out_npy = os.path.join(out_folder, out_base + ".npy")
        out_meta = os.path.join(out_folder, out_base + "_meta.json")
        np.save(out_npy, out_arr)

        meta = {
            "orig_epoch_file": str(p),
            "target_channels": TARGET_CHANS,
            "n_channels": int(out_arr.shape[0]),
            "n_samples": int(out_arr.shape[1])
        }
        if big_meta:
            meta.update({"orig_meta": big_meta})
            if "label" in big_meta: meta["label"] = big_meta["label"]
        json.dump(meta, open(out_meta, "w"), indent=2)
        print("Saved standardized:", out_npy, "shape=", out_arr.shape)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python select_32_channels.py <input_epoch_folder> <output_folder>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
