# extract_features.py
import os, json, numpy as np
from scipy.signal import welch, hilbert
from itertools import combinations
from pathlib import Path

# frequency bands (Hz)
BANDS = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta": (13,30)
}

def bandpower_from_psd(f, Pxx, lo, hi):
    # integrate PSD over band using simple sum * df
    df = f[1] - f[0]
    idx = (f >= lo) & (f <= hi)
    return np.sum(Pxx[idx]) * df

def compute_psd_epoch(epoch, sfreq):
    # epoch shape: (channels, samples)
    nchan = epoch.shape[0]
    band_powers = np.zeros((nchan, len(BANDS)))
    f, _ = welch(epoch[0], fs=sfreq, nperseg=min(2048, int(sfreq*2)))
    # compute per-channel PSD and band powers
    for ch in range(nchan):
        f, Pxx = welch(epoch[ch], fs=sfreq, nperseg=min(2048, int(sfreq*2)))
        for i,(bn,(lo,hi)) in enumerate(BANDS.items()):
            band_powers[ch,i] = bandpower_from_psd(f, Pxx, lo, hi)
    # log transform (avoid log(0))
    band_powers = np.log10(band_powers + 1e-12)
    return band_powers  # shape (nchan, nbands)

def compute_plv(epoch):
    # epoch: (channels, samples)
    analytic = hilbert(epoch, axis=1)
    phases = np.angle(analytic)  # (ch, samples)
    nchan = phases.shape[0]
    pairs = list(combinations(range(nchan), 2))
    plv_vals = []
    for (i,j) in pairs:
        pd = phases[i] - phases[j]
        plv = np.abs(np.mean(np.exp(1j * pd)))
        plv_vals.append(plv)
    return np.array(plv_vals)  # shape (npairs,)

def process_folder(in_folder, out_folder, sfreq_from_meta=True):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted(Path(in_folder).glob("*_32ch.npy"))
    if not files:
        print("No standardized epoch files found in", in_folder)
        return
    for f in files:
        base = f.stem
        meta_path = str(f).replace("_32ch.npy", "_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
            except:
                meta = {}
        sfreq = meta.get("orig_meta", {}).get("sfreq") or meta.get("sfreq") or meta.get("orig_meta",{}).get("srate")
        if sfreq is None:
            # try reading sample length and assume 1 Hz (shouldn't happen)
            sfreq = 250.0
        else:
            sfreq = float(sfreq)
        epoch = np.load(f)  # shape (32, samples)
        psd = compute_psd_epoch(epoch, sfreq)       # (32, 4)
        plv = compute_plv(epoch)                   # (32*31/2,)
        out_npz = os.path.join(out_folder, base + "_features.npz")
        label = meta.get("label", meta.get("orig_meta",{}).get("label","unknown"))
        np.savez_compressed(out_npz, psd=psd, plv=plv, label=label, meta=meta)
        print("Saved features:", out_npz, "psd.shape", psd.shape, "plv.shape", plv.shape, "label", label)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_features.py <in_standardized_folder> <out_feature_folder>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
