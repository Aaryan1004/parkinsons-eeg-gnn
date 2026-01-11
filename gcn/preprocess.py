import os
import mne

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "data"        # where your raw .vhdr/.eeg/.vmrk files are stored
OUTPUT_DIR = "processed" # where cleaned EDFs will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess_brainvision(vhdr_file):
    print(f"\n=== Processing {vhdr_file} ===")

    # Load BrainVision file
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
    print(raw.info)

    # Step 1: Bandpass filter (keep 0.5–40 Hz)
    raw.filter(0.5, 40., fir_design="firwin")

    # Step 2: Notch filter to remove line noise (50 Hz in India/Europe, change to 60 if US)
    raw.notch_filter(freqs=[50])

    # Step 3: ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter="auto")
    ica.fit(raw)

    # (Optional) Let user manually inspect ICA components
    # ica.plot_components()  # uncomment if you want interactive plots

    # Apply ICA
    raw_clean = ica.apply(raw.copy())

    # Step 4: Export to EDF
    base_name = os.path.splitext(os.path.basename(vhdr_file))[0]
    out_file = os.path.join(OUTPUT_DIR, f"{base_name}_clean.edf")

    mne.export.export_raw(
        out_file,
        raw_clean,
        fmt="edf",
        physical_range=(-200e-6, 200e-6)  # typical EEG scaling
    )

    print(f"✅ Saved cleaned EDF: {out_file}")
    return out_file


# ----------------------------
# Batch process all .vhdr files
# ----------------------------
if __name__ == "__main__":
    vhdr_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".vhdr")]
    if len(vhdr_files) == 0:
        print(f"No .vhdr files found in {DATA_DIR}")
    else:
        for file in vhdr_files:
            vhdr_path = os.path.join(DATA_DIR, file)
            preprocess_brainvision(vhdr_path)
