# scan_labels.py
import glob, os, numpy as np
from dataset_loader import _sanitize_label_raw, LABEL_MAP, infer_label_from_meta, load_feature_file

folders = ["features/Iowa","features/UC","features/UNM_on","features/UNM_off"]
for folder in folders:
    paths = sorted(glob.glob(os.path.join(folder, "*_features.npz")))
    labeled = 0
    unlabeled = 0
    for f in paths:
        try:
            psd, plv, raw_label, meta = load_feature_file(f)
            lab = _sanitize_label_raw(raw_label, label_map=LABEL_MAP if 'LABEL_MAP' in globals() else None)
            if lab is None:
                lab = _sanitize_label_raw(infer_label_from_meta(meta))
            if lab is None:
                bn = os.path.basename(f).lower()
                if "pd" in bn or "patient" in bn:
                    lab = 1
                elif "control" in bn or "ctrl" in bn or "hc" in bn:
                    lab = 0
            if lab is None:
                unlabeled += 1
            else:
                labeled += 1
        except Exception:
            unlabeled += 1
    print(f"{folder} → total: {len(paths)}, labeled: {labeled}, unlabeled/skipped: {unlabeled}")
