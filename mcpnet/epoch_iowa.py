import os
import numpy as np
import json

def extract_epochs(data, sfreq, epoch_seconds=32):
    samples_per_epoch = int(epoch_seconds * sfreq)
    total_samples = data.shape[1]

    epochs = []
    for start in range(0, total_samples - samples_per_epoch + 1, samples_per_epoch):
        end = start + samples_per_epoch
        epochs.append(data[:, start:end])
    return epochs

def process_file(npy_path, meta_path, out_dir, epoch_seconds=32):

    print(f"\nProcessing {npy_path}")

    data = np.load(npy_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    sfreq = meta['sfreq']

    epochs = extract_epochs(data, sfreq, epoch_seconds)

    base = os.path.splitext(os.path.basename(npy_path))[0]
    for i, ep in enumerate(epochs):
        out_path = os.path.join(out_dir, f"{base}_epoch{i+1}.npy")
        np.save(out_path, ep)

    print(f"Saved {len(epochs)} epochs to {out_dir}")


def main():
    in_dir = "Data/Iowa/epochs_iowa"
    out_dir = "Data/Iowa/epochs_iowa_segmented"

    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]

    for f in files:
        npy_path = os.path.join(in_dir, f)
        meta_path = npy_path.replace(".npy", "_meta.json")

        process_file(npy_path, meta_path, out_dir, epoch_seconds=32)

    print("\nFinished epoching Iowa dataset!")


if __name__ == "__main__":
    main()
