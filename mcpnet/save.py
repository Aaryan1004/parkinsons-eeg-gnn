# verify_features.py
import glob, os
from dataset_loader_common import load_feature_file, infer_label_from_filename, infer_label_from_meta

FOLDERS = ["features/Iowa", "features/UC"]   # mirror FEATURE_FOLDERS in train script

all_files = []
for fld in FOLDERS:
    files = sorted(glob.glob(os.path.join(fld, "*_features.npz")))
    print(f"\nFolder: {fld}  -> {len(files)} files")
    all_files.extend([(fld, f) for f in files])

# show first 10 examples total with label/meta
print("\nFirst 20 examples (folder, path, inferred_label_from_meta, inferred_label_from_filename):")
for i, (fld, p) in enumerate(all_files[:20]):
    try:
        psd, plv, label, meta = load_feature_file(p)
    except Exception as e:
        print(i, fld, os.path.basename(p), "load_error:", e)
        continue
    lbl_meta = infer_label_from_meta(meta)
    lbl_fname = infer_label_from_filename(p)
    print(i, fld, os.path.basename(p), "meta_label:", repr(label), "inferred_meta:", lbl_meta, "inferred_fname:", lbl_fname)
