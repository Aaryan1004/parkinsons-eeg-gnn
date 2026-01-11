import os
import sys
import json
import numpy as np

def extract_epochs(signal, fs, epoch_seconds=32):
    epoch_len = fs * epoch_seconds
    n_samples = signal.shape[1]

    epochs = []
    start = 0
    while start + epoch_len <= n_samples:
        segment = signal[:, start:start+epoch_len]
        epochs.append(segment)
        start += epoch_len  # non-overlapping

    return epochs

def main():
    if len(sys.argv) < 4:
        print("Usage: python select_32_and_epoch.py <input_npy> <input_json> <output_folder> [--target_seconds 32]")
        sys.exit(1)

    npy_file = sys.argv[1]
    json_file = sys.argv[2]
    out_dir  = sys.argv[3]

    target_seconds = 32
    if "--target_seconds" in sys.argv:
        idx = sys.argv.index("--target_seconds")
        target_seconds = int(sys.argv[idx+1])

    os.makedirs(out_dir, exist_ok=True)

    # Load data
    data = np.load(npy_file)
    with open(json_file, "r") as f:
        meta = json.load(f)

    fs = meta.get("srate", 1000)     # UNM uses 1000 Hz
    n_channels = data.shape[0]

    print(f"Loaded {npy_file}: shape={data.shape}, fs={fs}")

    epochs = extract_epochs(data, fs, target_seconds)
    print(f"Extracted {len(epochs)} epochs of {target_seconds} seconds")

    # Save epochs
    base = os.path.splitext(os.path.basename(npy_file))[0]
    for i, ep in enumerate(epochs):
        out_path = os.path.join(out_dir, f"{base}_epoch{i+1}.npy")
        np.save(out_path, ep)

    print(f"Saved all epochs to {out_dir}")

if __name__ == "__main__":
    main()
