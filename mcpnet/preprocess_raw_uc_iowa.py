# preprocess_raw_uc_iowa.py
import sys, os, json, numpy as np, mne

def load_any_raw(path):
    ext = path.split('.')[-1].lower()
    if ext == "vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    elif ext == "edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == "bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif ext == "set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    else:
        raise RuntimeError("Unsupported extension: "+ext)
    return raw

def preprocess_raw(raw, l_freq=0.5, h_freq=50.0):
    raw.load_data()
    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
    # notch at mains (50 Hz) — use 50 Hz (India)
    raw.notch_filter(freqs=[50.0], verbose=False)
    # run ICA to remove EOG if possible
    try:
        n_comp = min(20, raw.info['nchan'] - 1)
        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42, max_iter='auto')
        ica.fit(raw, decim=5, verbose=False)
        # try to find EOG components (if EOG channel exists)
        eog_inds, scores = ica.find_bads_eog(raw, threshold=3.0, verbose=False) if 'EOG' in ''.join(raw.ch_names).upper() else ([], [])
        if eog_inds:
            ica.exclude = eog_inds
            ica.apply(raw)
    except Exception as e:
        # ICA can fail for low rank or small channel counts — continue with filtered data
        print("ICA failed/skip:", e)
    return raw

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_raw_uc_iowa.py <input_raw_file> <output_npy_fullpath>")
        sys.exit(1)

    inp = sys.argv[1]
    outnpy = sys.argv[2]
    outmeta = outnpy.replace('.npy','_meta.json')

    raw = load_any_raw(inp)
    raw = preprocess_raw(raw)
    arr = raw.get_data()  # shape channels x samples
    np.save(outnpy, arr)

    meta = {
        "channel_names": raw.ch_names,
        "srate": int(raw.info['sfreq']),
        "n_channels": int(raw.info['nchan'])
    }
    with open(outmeta,'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved cleaned:", outnpy, "shape", arr.shape)
