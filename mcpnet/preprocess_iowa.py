import os
import mne
import numpy as np
import json

def process_one_subject(vhdr_path, out_dir):

    print(f"\nProcessing {vhdr_path}")

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

    # Basic Info
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names']

    # ----- Filtering -----
    raw.filter(l_freq=1.0, h_freq=45.0)

    # Get numpy array (channels × samples)
    data = raw.get_data()

    # Prepare output paths
    base = os.path.splitext(os.path.basename(vhdr_path))[0]
    out_npy = os.path.join(out_dir, base + ".npy")
    out_meta = os.path.join(out_dir, base + "_meta.json")

    # Save data
    np.save(out_npy, data)

    # Metadata
    meta = {
        "sfreq": sfreq,
        "ch_names": ch_names,
        "n_channels": data.shape[0],
        "n_samples": data.shape[1]
    }

    with open(out_meta, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {out_npy}, shape={data.shape}")
    print(f"Saved {out_meta}")


def main():
    in_dir = "Data/Iowa/raw"
    out_dir = "Data/Iowa/epochs_iowa"

    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if file.endswith(".vhdr"):
            vhdr_path = os.path.join(in_dir, file)
            process_one_subject(vhdr_path, out_dir)

    print("\nAll Iowa subjects processed!")


if __name__ == "__main__":
    main()
