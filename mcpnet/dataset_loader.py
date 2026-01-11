# dataset_loader.py (updated: handle 'unknown' labels and filename heuristics)
import os, glob, re, numpy as np
from torch.utils.data import Dataset

LABEL_MAP = {
    "PD": 1, "pd": 1, "1": 1, 1: 1,
    "HC": 0, "hc": 0, "0": 0, 0: 0,
    "CONTROL": 0, "control": 0, "ctl": 0
}

def infer_label_from_meta(meta_obj):
    if meta_obj is None:
        return None
    try:
        if isinstance(meta_obj, np.ndarray) and meta_obj.size == 1:
            return infer_label_from_meta(meta_obj.item())
    except Exception:
        pass
    if isinstance(meta_obj, (int, float)):
        return int(meta_obj)
    if isinstance(meta_obj, str):
        s = meta_obj.strip()
        if not s:
            return None
        ls = s.lower()
        if ls in LABEL_MAP:
            return LABEL_MAP[ls]
        # treat explicit 'unknown' as missing
        if ls == "unknown":
            return None
        try:
            return int(s)
        except:
            pass
        if any(k in ls for k in ["parkin", "pd", "patient"]):
            return 1
        if any(k in ls for k in ["hc", "control", "healthy"]):
            return 0
    if isinstance(meta_obj, dict):
        for k in ("label","diagnosis","group","y","target"):
            if k in meta_obj:
                return infer_label_from_meta(meta_obj[k])
        if "orig_meta" in meta_obj:
            return infer_label_from_meta(meta_obj["orig_meta"])
    return None

def load_feature_file(path):
    """
    Loads .npz and returns (psd_array, plv_array, label, meta)
    """
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"np.load failed: {e}")
    files = getattr(z, "files", None)
    if files is not None:
        psd = z.get('psd', None)
        plv = z.get('plv', None)
        label = z.get('label', None)
        meta = z.get('meta', None)
        # unwrap 0-d object arrays
        for name, val in (('psd', psd), ('plv', plv), ('label', label), ('meta', meta)):
            if isinstance(val, np.ndarray) and val.size == 1:
                try:
                    v = val.item()
                    if name == 'psd': psd = v
                    elif name == 'plv': plv = v
                    elif name == 'label': label = v
                    elif name == 'meta': meta = v
                except Exception:
                    pass
        if psd is None or plv is None:
            raise RuntimeError(f"missing psd/plv in npz keys {list(z.files)}")
        return np.array(psd), np.array(plv), label, meta

    # fallback: ndarray containing dict
    if isinstance(z, np.ndarray) and z.size == 1:
        try:
            obj = z.item()
            if isinstance(obj, dict):
                return np.array(obj.get('psd')), np.array(obj.get('plv')), obj.get('label', None), obj.get('meta', None)
        except Exception as e:
            raise RuntimeError(f"unexpected ndarray contents: {e}")

    raise RuntimeError("Unrecognized .npz structure")

def is_constant(arr, atol=1e-14, rtol=1e-6):
    """
    True if array is practically constant.
    Uses absolute + relative tolerance so very small-but-varying arrays are NOT treated as constant.
    """
    a = np.array(arr).reshape(-1)
    if a.size == 0:
        return True
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if np.isnan(mn) or np.isnan(mx):
        return True
    # absolute check
    if (mx - mn) <= atol:
        return True
    # relative check vs mean absolute magnitude
    mean_abs = max(atol, float(np.nanmean(np.abs(a))))
    if (mx - mn) <= (rtol * mean_abs):
        return True
    return False

def infer_label_from_filename(path):
    """
    Simple substring heuristics from filename/folder.
    Returns 0/1 or None.
    """
    bn = os.path.basename(path).lower()
    dn = os.path.basename(os.path.dirname(path)).lower()
    checks = [bn, dn]
    for s in checks:
        if 'control' in s or 'ctl' in s or 'hc' in s or 'healthy' in s:
            return 0
        if 'pd' in s or 'parkin' in s or 'patient' in s or 'sub-pd' in s or s.startswith('pd'):
            return 1
    # also check patterns like 'sub-hc10' or 'sub-hc'
    if re.search(r'sub-?hc', bn) or re.search(r'sub_?hc', bn):
        return 0
    if re.search(r'sub-?pd', bn) or re.search(r'sub_?pd', bn):
        return 1
    return None

class EEGFeatureDataset(Dataset):
    def __init__(self, folders, transform=None, require_labels=True, skip_constant=True, debug_limit_examples=5):
        if isinstance(folders, str):
            folders = [folders]
        self.files = []
        for d in folders:
            self.files += glob.glob(os.path.join(d, "*_features.npz"))
        self.files = sorted(self.files)
        self.transform = transform
        self.items = []

        # debug counters
        self._cnt_unreadable = 0
        self._cnt_constant = 0
        self._cnt_unlabeled = 0
        self._examples_unlabeled = []
        self._examples_constant = []
        self._examples_unreadable = []

        for f in self.files:
            try:
                psd, plv, label, meta = load_feature_file(f)
            except Exception as e:
                self._cnt_unreadable += 1
                if len(self._examples_unreadable) < debug_limit_examples:
                    self._examples_unreadable.append((f, str(e)))
                continue

            # skip constant if requested
            if skip_constant and (is_constant(psd) or is_constant(plv)):
                self._cnt_constant += 1
                if len(self._examples_constant) < debug_limit_examples:
                    try:
                        r = float(np.nanmax(psd) - np.nanmin(psd))
                    except Exception:
                        r = None
                    self._examples_constant.append((f, r))
                continue

            # If label is explicit string 'unknown' treat as missing
            if isinstance(label, str) and label.strip().lower() == "unknown":
                label = None

            # infer label
            if label is None:
                label = infer_label_from_meta(meta)
            if label is None:
                label = infer_label_from_filename(f)

            # final fallback: try to parse numeric label strings
            if label is not None and isinstance(label, str):
                s = label.strip().lower()
                if s in LABEL_MAP:
                    label = LABEL_MAP[s]
                else:
                    try:
                        label = int(s)
                    except:
                        pass

            if label is None and require_labels:
                self._cnt_unlabeled += 1
                if len(self._examples_unlabeled) < debug_limit_examples:
                    self._examples_unlabeled.append((f, meta))
                continue

            # append valid
            try:
                self.items.append((f, int(label)))
            except Exception:
                self._cnt_unlabeled += 1
                if len(self._examples_unlabeled) < debug_limit_examples:
                    self._examples_unlabeled.append((f, label))
                continue

        # Summary debug prints
        print(f"Dataset loader summary for folders: {folders}")
        print(f"  total files discovered: {len(self.files)}")
        print(f"  kept items: {len(self.items)}")
        print(f"  skipped - unreadable: {self._cnt_unreadable}, constant: {self._cnt_constant}, unlabeled: {self._cnt_unlabeled}")

        if len(self._examples_constant):
            print("  examples skipped-as-constant (path, approx_range):")
            for p,r in self._examples_constant:
                print("   ", p, r)
        if len(self._examples_unlabeled):
            print("  examples skipped-unlabeled (path, meta/label):")
            for p,m in self._examples_unlabeled:
                print("   ", p, m)
        if len(self._examples_unreadable):
            print("  examples unreadable (path, error):")
            for p,e in self._examples_unreadable:
                print("   ", p, e)

        if len(self.items) == 0:
            raise RuntimeError("No labeled feature files found in: " + ", ".join(folders))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        psd, plv, _, _ = load_feature_file(path)
        psd = np.array(psd, dtype=np.float32).reshape(-1)
        plv = np.array(plv, dtype=np.float32).reshape(-1)
        if self.transform:
            psd, plv = self.transform(psd, plv)
        return psd, plv, int(label)

    def class_counts(self):
        c0 = sum(1 for _,l in self.items if l==0)
        c1 = sum(1 for _,l in self.items if l==1)
        return {0:c0, 1:c1}

    def subjects_map(self):
        mp = {}
        for path,label in self.items:
            bn = os.path.basename(path)
            base = bn.split('_epoch')[0]
            sub = base.lower()
            mp.setdefault(sub, []).append((path,label))
        return mp
