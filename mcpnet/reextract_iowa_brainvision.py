# reextract_iowa_brainvision.py
# pip install mne numpy scipy
import os, glob
import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert
import mne

# ---------- EDIT IF NEEDED ----------
RAW_DIR = "Data/Iowa/raw"     # where your .vhdr/.vmrk/.eeg files live
OUT_DIR = "features/Iowa"     # where to save extracted .npz (will be created)
PICK_N_CHANNELS = 32
EPOCH_SECONDS = 10
BANDS = [(1,4),(4,8),(8,13),(13,30)]
PLV_BAND = (8,13)
NPERSEG = 512
RESAMPLE_TO = None            # set to 256 if you want uniform sr
VERBOSE = True
# ------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def compute_psd_welch(ch_data, fs, nperseg=NPERSEG, bands=BANDS):
    nperseg = min(nperseg, ch_data.shape[1])
    psd_feats = []
    for ch in ch_data:
        f, P = welch(ch, fs=fs, nperseg=nperseg)
        band_powers = [(P[(f>=a)&(f<=b)].mean() if ((f>=a)&(f<=b)).sum()>0 else 0.0) for (a,b) in bands]
        psd_feats.append(band_powers)
    return np.array(psd_feats)   # (C, n_bands)

def compute_plv(ch_data, fs, band=PLV_BAND):
    if ch_data.shape[1] < 10:
        C = ch_data.shape[0]
        return np.zeros((C*(C-1))//2, dtype=float)
    b, a = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
    analytic = []
    for ch in ch_data:
        try:
            x = filtfilt(b, a, ch)
        except Exception:
            x = ch
        analytic.append(np.angle(hilbert(x)))
    analytic = np.array(analytic)
    C = analytic.shape[0]
    plv = []
    for i in range(C):
        for j in range(i+1, C):
            diff = analytic[i] - analytic[j]
            plv.append(np.abs(np.exp(1j * diff).mean()))
    return np.array(plv)

def is_constant(arr):
    a = np.array(arr).reshape(-1)
    return np.ptp(a) == 0

vhdr_files = glob.glob(os.path.join(RAW_DIR, "*.vhdr"))
if len(vhdr_files) == 0:
    raise RuntimeError(f"No .vhdr files found in {RAW_DIR}")

for vhdr in sorted(vhdr_files):
    try:
        if VERBOSE: print("\nReading:", vhdr)
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
        if RESAMPLE_TO is not None:
            raw.resample(RESAMPLE_TO, npad="auto")
        fs = raw.info['sfreq']
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')
        if len(picks) == 0:
            print("  WARNING: no EEG channels found, skipping", vhdr); continue
        picks = picks[:min(len(picks), PICK_N_CHANNELS)]
        if VERBOSE: print(f"  picked {len(picks)} EEG channels, fs={fs}")

        # events -> windows around events; fallback to sliding windows
        events, event_ids = mne.events_from_annotations(raw, verbose=False)
        windows = []
        if len(events) > 0:
            for ev in events:
                s = int(ev[0])
                e = s + int(EPOCH_SECONDS * fs)
                if e <= raw.n_times:
                    windows.append((s, e))
        if len(windows) == 0:
            step = int(EPOCH_SECONDS * fs)
            n_epochs = raw.n_times // step
            windows = [(k*step, k*step + step) for k in range(n_epochs)]

        base = os.path.splitext(os.path.basename(vhdr))[0]
        saved = 0
        for idx, (samp_s, samp_e) in enumerate(windows, 1):
            data, _ = raw[picks, samp_s:samp_e]  # shape (C, samples)
            if data.shape[1] < 4: 
                continue
            psd = compute_psd_welch(data, fs=fs)
            plv = compute_plv(data, fs=fs, band=PLV_BAND)
            if is_constant(psd) or is_constant(plv) or np.isnan(psd).any() or np.isnan(plv).any():
                if VERBOSE: print(f"  SKIP epoch {idx} (bad) psd_const={is_constant(psd)} plv_const={is_constant(plv)}")
                continue
            outname = f"{base}_epoch{idx}_32ch_features.npz"
            outpath = os.path.join(OUT_DIR, outname)
            np.savez(outpath, psd=psd, plv=plv, label='unknown', meta={})
            saved += 1
        print(f"  finished {base}, saved {saved} epochs -> {OUT_DIR}")

    except Exception as e:
        print("ERR processing", vhdr, e)

print("\nDone. Re-extracted features saved in", OUT_DIR)
