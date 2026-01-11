# preprocess_uc.py
import os, json, sys
import numpy as np
import mne
from glob import glob

# heuristics for inferring label from path/name
PD_KEYWORDS = ["pd", "parkin", "patient", "parkinson"]
HC_KEYWORDS = ["hc", "control", "ctl", "healthy"]

def infer_label_from_path(path):
    low = path.lower()
    # check folder names & file names
    for k in PD_KEYWORDS:
        if k in low:
            return "PD"
    for k in HC_KEYWORDS:
        if k in low:
            return "HC"
    return None

def find_beh_label(subject_root):
    """
    Try to find behavior TSV/JSON that contains diagnostic info.
    This will scan subject_root for any .tsv or .json files and
    attempt a naive search for 'diagnosis' or 'group' tokens.
    """
    for ext in ("*.tsv","*.json"):
        for f in glob(os.path.join(subject_root, ext)):
            try:
                if f.endswith(".tsv"):
                    import pandas as pd
                    df = pd.read_csv(f, sep="\t")
                    for col in df.columns:
                        if "diagnos" in col.lower() or "group" in col.lower() or "label" in col.lower():
                            # take first non-null
                            val = df[col].dropna().astype(str).iloc[0]
                            if any(k in val.lower() for k in PD_KEYWORDS):
                                return "PD"
                            if any(k in val.lower() for k in HC_KEYWORDS):
                                return "HC"
                else:
                    with open(f, "r", encoding="utf8") as fh:
                        txt = fh.read().lower()
                        if any(k in txt for k in PD_KEYWORDS):
                            return "PD"
                        if any(k in txt for k in HC_KEYWORDS):
                            return "HC"
            except Exception:
                continue
    return None

def load_any_raw(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    else:
        raise RuntimeError("Unsupported extension: "+ext)
    return raw

def preprocess_raw(raw, l_freq=1.0, h_freq=45.0):
    raw.load_data()
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    # notch at mains (50Hz) — adjust to 60 if your data uses that
    raw.notch_filter(freqs=[50.0], verbose=False)
    # try simple ICA removal if reasonable
    try:
        nchan = raw.info.get("nchan", raw._data.shape[0])
        n_comp = min(20, max(1, nchan - 1))
        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42, max_iter="auto")
        ica.fit(raw, decim=5, verbose=False)
        # try to auto-detect EOG components if EOG channels exist
        eog_inds, scores = ([], [])
        try:
            eog_inds, scores = ica.find_bads_eog(raw, verbose=False)
        except Exception:
            pass
        if eog_inds:
            ica.exclude = eog_inds
            ica.apply(raw)
    except Exception as e:
        # ICA may fail — that's fine, continue
        print("ICA skipped/failed:", str(e))
    return raw

def process_file(in_path, out_dir):
    print(f"\nLoading {in_path}")
    raw = load_any_raw(in_path)
    # infer label using path heuristics
    label = infer_label_from_path(in_path)
    if not label:
        # try parent folder level up & subject root
        parent = os.path.dirname(in_path)
        label = infer_label_from_path(parent)
        if not label:
            # try folder two levels up (BIDS: sub-xx/ses-yy/eeg)
            label = infer_label_from_path(os.path.dirname(parent))
    # also try any beh metadata near this file
    beh_label = find_beh_label(os.path.dirname(in_path))
    if beh_label:
        label = beh_label

    raw = preprocess_raw(raw, l_freq=1.0, h_freq=45.0)
    data = raw.get_data()  # channels x samples

    base = os.path.splitext(os.path.basename(in_path))[0]
    out_npy = os.path.join(out_dir, base + "_cleaned.npy")
    out_meta = os.path.join(out_dir, base + "_meta.json")

    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "ch_names": raw.info["ch_names"],
        "n_channels": int(raw.info["nchan"]),
        "n_samples": int(data.shape[1]),
        "label": label if label else "unknown",
        "orig_path": os.path.abspath(in_path)
    }

    np.save(out_npy, data)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_npy, "shape=", data.shape, "label=", meta["label"])

def main():
    raw_root = os.path.join("Data", "UC", "raw")
    if not os.path.exists(raw_root):
        print("Expected UC raw folder at:", raw_root)
        sys.exit(1)

    out_dir = os.path.join("Data", "UC", "cleaned")
    os.makedirs(out_dir, exist_ok=True)

    # find EEG files recursively
    patterns = ["**/*.bdf","**/*.edf","**/*.vhdr","**/*.set"]
    found = []
    for pat in patterns:
        found.extend(glob(os.path.join(raw_root, pat), recursive=True))

    found = sorted(set(found))
    if not found:
        print("No raw files found under", raw_root)
        return

    print("Found", len(found), "files. Processing...")

    for f in found:
        try:
            process_file(f, out_dir)
        except Exception as e:
            print("FAILED to process", f, ":", str(e))

    print("\nAll UC files processed. Cleaned files are in:", out_dir)

if __name__ == "__main__":
    main()
