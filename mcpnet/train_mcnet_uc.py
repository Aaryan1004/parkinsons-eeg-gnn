"""
Train MCNet on UC features using LOSO (leave-one-subject-out).

Place this file next to dataset_loader.py and mcnet_model.py.
Change FEATURE_FOLDERS to point to your UC features folder(s).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from dataset_loader import EEGFeatureDataset   # your loader
from train_mcnet_iowa import MCNet

# ---------- CONFIG ----------
FEATURE_FOLDERS = ["features/UC"]   # <- change if needed (e.g. ["features/Iowa"])
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
SAVE_DIR = "models_uc"
SKIP_CONSTANT = False   # set True if you want to skip constant-feature files
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------- Build dataset ----------
print("Building dataset from:", FEATURE_FOLDERS)
try:
    # dataset loader should accept (folders, transform=None, skip_constant=False)
    dataset = EEGFeatureDataset(FEATURE_FOLDERS, transform=None, skip_constant=SKIP_CONSTANT)
except TypeError:
    # fallback if your dataset constructor signature differs
    dataset = EEGFeatureDataset(FEATURE_FOLDERS)

n_total = len(dataset)
print("Total samples:", n_total)
if n_total == 0:
    raise RuntimeError("No samples in dataset. Check FEATURE_FOLDERS and dataset_loader output.")

# print class distribution
try:
    counts = dataset.class_counts()
    print("Class counts (0=HC,1=PD):", counts)
except Exception:
    # fallback if dataset doesn't expose class_counts()
    labels = [lbl for _, _, lbl in dataset]
    unique, counts_arr = np.unique(labels, return_counts=True)
    counts = dict(zip(map(int, unique), map(int, counts_arr)))
    print("Class counts (fallback):", counts)

# get dims from first example
psd_example, plv_example, lbl = dataset[0]
psd_dim = int(psd_example.numel())
plv_dim = int(plv_example.numel())
print("psd_dim", psd_dim, "plv_dim", plv_dim)

# ---------- LOSO subjects ----------
# dataset must expose per-sample metadata subject id OR dataset.get_subjects()
if hasattr(dataset, "subjects_per_sample"):
    # if dataset provided a mapping
    subj_list = sorted(set(dataset.subjects_per_sample))
else:
    # try to derive subject ids from dataset.items or dataset._items
    subj_list = None

# Best effort: get list of unique subject ids from dataset (dataset should include meta)
subjects = []
try:
    for idx in range(len(dataset)):
        sample_meta = None
        try:
            # many dataset implementations return (psd,plv,label,meta)
            sample = dataset[idx]
            if len(sample) == 4:
                _, _, _, meta = sample
                sample_meta = meta
        except Exception:
            pass

        # fallback: dataset might expose items or item_meta array
        if sample_meta is None:
            continue

        sid = None
        # common keys
        if isinstance(sample_meta, dict):
            sid = sample_meta.get("subject") or sample_meta.get("subj") or sample_meta.get("id")
        else:
            # if meta is a string subject id
            sid = sample_meta

        if sid:
            subjects.append(str(sid))
except Exception:
    pass

# last-chance: try dataset.subjects() or dataset.get_subjects()
if len(subjects) == 0:
    if hasattr(dataset, "get_subjects"):
        subjects = list(map(str, dataset.get_subjects()))
    elif hasattr(dataset, "subjects"):
        subjects = list(map(str, dataset.subjects))
    else:
        # if none available, fallback to a single fold training
        subjects = ["__all__"]

subjects = sorted(list(set(subjects)))
print("Detected subjects (for LOSO):", len(subjects))
# if only "__all__", we'll do k-fold like holdout
if subjects == ["__all__"]:
    do_loso = False
else:
    do_loso = True

# ---------- training helpers ----------
def train_one_fold(train_indices, val_indices, fold_name):
    # create subset datasets
    class Subset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.dataset[self.indices[i]]

    train_ds = Subset(dataset, train_indices)
    val_ds   = Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MCNet(psd_dim, plv_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        t0 = time.time()
        for psd, plv, labels in train_loader:
            psd = psd.view(psd.size(0), -1).to(DEVICE).float()
            plv = plv.view(plv.size(0), -1).to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(psd, plv)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for psd, plv, labels in val_loader:
                psd = psd.view(psd.size(0), -1).to(DEVICE).float()
                plv = plv.view(plv.size(0), -1).to(DEVICE).float()
                labels = labels.to(DEVICE).long()

                outputs = model(psd, plv)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true)>0 else 0.0
        print(f"{fold_name} epoch {epoch}/{EPOCHS} loss={np.mean(losses):.4f} val_acc={acc:.4f} val_f1={f1:.4f} time={(time.time()-t0):.1f}s")

        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": acc,
                "val_f1": f1
            }
            fname = os.path.join(SAVE_DIR, f"best_{fold_name}_acc_{acc:.4f}.pth")
            torch.save(best_state, fname)

    return best_val_acc

# ---------- Run LOSO or single-fold ----------
overall_best = -1.0
if do_loso and subjects and subjects[0] != "__all__":
    # build index lists per subject using dataset metadata.
    # we try several common interfaces dataset_loader may provide.
    idx_to_subject = []
    for idx in range(len(dataset)):
        try:
            entry = dataset._items[idx]   # if dataset stores items internally
            # entry might be (path, label, meta) or (psd,plv,label,meta)
            if isinstance(entry, tuple) and len(entry) >= 3:
                meta = entry[-1]
                sid = None
                if isinstance(meta, dict):
                    sid = meta.get("subject") or meta.get("subj") or meta.get("id")
                else:
                    sid = meta
            else:
                sid = None
        except Exception:
            # fallback: dataset may return psd,plv,label,meta via dataset[idx]
            try:
                sample = dataset[idx]
                if len(sample) == 4:
                    _, _, _, meta = sample
                    if isinstance(meta, dict):
                        sid = meta.get("subject") or meta.get("subj") or meta.get("id")
                    else:
                        sid = meta
                else:
                    sid = None
            except Exception:
                sid = None

        idx_to_subject.append(str(sid) if sid is not None else "__unknown__")

    unique_subjects = sorted(list(set(idx_to_subject)))
    print("Subjects discovered from dataset items:", len(unique_subjects))

    # for each subject, do LOSO
    for fold_idx, test_subj in enumerate(unique_subjects, start=1):
        # skip unknown subjects if they dominate
        if test_subj == "__unknown__":
            print("Skipping unknown-subject fold")
            continue

        test_indices = [i for i,s in enumerate(idx_to_subject) if s == test_subj]
        train_indices = [i for i,s in enumerate(idx_to_subject) if s != test_subj]

        if len(test_indices) == 0 or len(train_indices) == 0:
            continue

        print(f"----- LOSO fold {fold_idx}/{len(unique_subjects)} test subject: {test_subj} -----")
        best_val = train_one_fold(train_indices, test_indices, fold_name=str(test_subj))
        print("Fold finished. Best val acc:", best_val)
        overall_best = max(overall_best, best_val)

else:
    # single-run (no LOSO)
    all_idx = list(range(len(dataset)))
    # shuffle and split train/val
    np.random.shuffle(all_idx)
    n_val = max(1, int(0.2 * len(all_idx)))
    val_idx = all_idx[:n_val]
    train_idx = all_idx[n_val:]
    print("Single run: train", len(train_idx), "val", len(val_idx))
    best_val = train_one_fold(train_idx, val_idx, fold_name="single")
    overall_best = max(overall_best, best_val)

print("ALL finished. Best val acc overall:", overall_best)
