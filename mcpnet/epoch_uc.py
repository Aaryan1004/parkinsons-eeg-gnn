# epoch_uc.py
import os, json, numpy as np, sys

def extract_epochs(data, sfreq, epoch_seconds=32):
    samples_per_epoch = int(epoch_seconds * sfreq)
    total = data.shape[1]
    epochs = []
    # non-overlapping fixed windows
    for start in range(0, total - samples_per_epoch + 1, samples_per_epoch):
        end = start + samples_per_epoch
        epochs.append(data[:, start:end])
    return epochs

def process_one(npy_path, meta_path, out_dir, epoch_seconds=32):
    data = np.load(npy_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    sfreq = meta.get("sfreq")
    if not sfreq:
        print("No srate in meta for", npy_path)
        return
    epochs = extract_epochs(data, sfreq, epoch_seconds=epoch_seconds)
    base = os.path.splitext(os.path.basename(npy_path))[0]
    label = meta.get("label","unknown")
    for i, ep in enumerate(epochs):
        out = os.path.join(out_dir, f"{base}_epoch{i+1}.npy")
        np.save(out, ep)
    # also save a small summary meta per file
    summary = {
        "n_epochs": len(epochs),
        "epoch_seconds": epoch_seconds,
        "sfreq": sfreq,
        "label": label
    }
    with open(os.path.join(out_dir, base + "_epoch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {len(epochs)} epochs to {out_dir} for {base} (label={label})")

def main():
    in_dir = os.path.join("Data", "UC", "cleaned")
    out_dir = os.path.join("Data", "UC", "epochs_uc")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(in_dir) if f.endswith("_cleaned.npy")])
    if not files:
        print("No cleaned .npy files found in", in_dir); return

    for f in files:
        npy_path = os.path.join(in_dir, f)
        meta_path = npy_path.replace("_cleaned.npy", "_meta.json")
        if not os.path.exists(meta_path):
            print("Missing meta for", f, "-> skipping")
            continue
        process_one(npy_path, meta_path, out_dir, epoch_seconds=32)

    print("UC epoching finished.")

if __name__ == "__main__":
    main()
