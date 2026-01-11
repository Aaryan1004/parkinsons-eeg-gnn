# dataset_loader_common.py
import os
import glob
import re
import numpy as np
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
    a = np.array(arr).reshape(-1)
    if a.size == 0:
        return True
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if np.isnan(mn) or np.isnan(mx):
        return True
    if (mx - mn) <= atol:
        return True
    mean_abs = max(atol, float(np.nanmean(np.abs(a))))
    if (mx - mn) <= (rtol * mean_abs):
        return True
    return False

def infer_label_from_filename(path):
    bn = os.path.basename(path).lower()
    dn = os.path.basename(os.path.dirname(path)).lower()
    checks = [bn, dn]
    for s in checks:
        if 'control' in s or 'ctl' in s or 'hc' in s or 'healthy' in s:
            return 0
        if 'pd' in s or 'parkin' in s or 'patient' in s or 'sub-pd' in s or s.startswith('pd'):
            return 1
    if re.search(r'sub-?hc', bn) or re.search(r'sub_?hc', bn):
        return 0
    if re.search(r'sub-?pd', bn) or re.search(r'sub_?pd', bn):
        return 1
    return None

class EEGFeatureDataset(Dataset):
    """
    A dataset that merges multiple feature folders.
    Items are stored as (fullpath, label, folder_prefix)
    """
    def __init__(self, folders, transform=None, require_labels=True, skip_constant=True, debug_limit_examples=5):
        if isinstance(folders, str):
            folders = [folders]
        self.folders = folders
        self.transform = transform
        self.items = []   # list of (path, label, subj_id)
        # debug
        self._cnt_unreadable = 0
        self._cnt_constant = 0
        self._cnt_unlabeled = 0
        self._examples_unlabeled = []
        self._examples_constant = []
        self._examples_unreadable = []

        discovered = []
        for folder in folders:
            pattern = os.path.join(folder, "*_features.npz")
            discovered += glob.glob(pattern)
        discovered = sorted(discovered)

        for f in discovered:
            try:
                psd, plv, label, meta = load_feature_file(f)
            except Exception as e:
                self._cnt_unreadable += 1
                if len(self._examples_unreadable) < debug_limit_examples:
                    self._examples_unreadable.append((f, str(e)))
                continue

            if skip_constant and (is_constant(psd) or is_constant(plv)):
                self._cnt_constant += 1
                if len(self._examples_constant) < debug_limit_examples:
                    try:
                        r = float(np.nanmax(psd) - np.nanmin(psd))
                    except Exception:
                        r = None
                    self._examples_constant.append((f, r))
                continue

            # normalize 'unknown' label
            if isinstance(label, str) and label.strip().lower() == "unknown":
                label = None

            # try meta first
            lbl = None
            lbl = infer_label_from_meta(meta) if meta is not None else None
            if lbl is None:
                lbl = infer_label_from_filename(f)
            # try direct label
            if lbl is None and label is not None:
                if isinstance(label, str):
                    s = label.strip().lower()
                    if s in LABEL_MAP:
                        lbl = LABEL_MAP[s]
                    else:
                        try:
                            lbl = int(s)
                        except:
                            lbl = None
                elif isinstance(label, (int, float)):
                    lbl = int(label)

            if lbl is None and require_labels:
                self._cnt_unlabeled += 1
                if len(self._examples_unlabeled) < debug_limit_examples:
                    self._examples_unlabeled.append((f, meta))
                continue

            # build subject id: try meta subject keys, else filename; prefix by folder to avoid collisions
            subj = None
            if isinstance(meta, dict):
                for k in ("subject", "subj", "id", "participant", "person"):
                    if k in meta:
                        subj = meta[k]
                        break
            if subj is None:
                # filename-level subject heuristic: before "_epoch"
                bn = os.path.basename(f)
                if "_epoch" in bn:
                    subj = bn.split("_epoch")[0]
                else:
                    subj = os.path.splitext(bn)[0]
            # prefix with folder name (last folder token) to avoid subject collisions across datasets
            folder_prefix = os.path.basename(os.path.normpath(os.path.dirname(f)))
            subj_id = f"{folder_prefix}__{str(subj).lower()}"

            try:
                self.items.append((f, int(lbl), subj_id))
            except Exception:
                self._cnt_unlabeled += 1
                if len(self._examples_unlabeled) < debug_limit_examples:
                    self._examples_unlabeled.append((f, lbl))
                continue

        # print summary
        print(f"Dataset loader summary for folders: {folders}")
        print(f"  total files discovered: {len(discovered)}")
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
        path, label, subj = self.items[idx]
        psd, plv, _, _ = load_feature_file(path)
        psd = np.array(psd, dtype=np.float32).reshape(-1)
        plv = np.array(plv, dtype=np.float32).reshape(-1)
        if self.transform:
            psd, plv = self.transform(psd, plv)
        return psd, plv, int(label), subj

    def class_counts(self):
        c0 = sum(1 for _,l,_ in self.items if l==0)
        c1 = sum(1 for _,l,_ in self.items if l==1)
        return {0:c0, 1:c1}

    def subjects_map(self):
        mp = {}
        for path,label,sub in self.items:
            mp.setdefault(sub, []).append((path,label))
        return mp

    def get_item_paths(self):
        return [p for p,_,_ in self.items]

    def get_items(self):
        return self.items
