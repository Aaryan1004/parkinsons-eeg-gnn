# train_test_common.py
"""
Train+Test script that works with dataset_loader_common.EEGFeatureDataset and mcnet_model.MCNet.

Usage:
    - Edit FEATURE_FOLDERS at top to include the folders you want (Iowa, UC, ...).
    - Set USE_LOSO=True to do leave-one-subject-out across combined subjects.
    - Or set USE_LOSO=False for a random train/val split.
    - For test-only, set TEST_ONLY=True and provide CHECKPOINT_PATH and optional TEST_FOLDERS.
"""

import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset_loader_common import EEGFeatureDataset, load_feature_file
from mcnet_model import MCNet

# ---------- CONFIG ----------
FEATURE_FOLDERS = ["features/UC"]   # <-- set the folders to merge
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
VAL_RATIO = 0.2
SAVE_DIR = "models_combined"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LOSO = True          # True => LOSO across combined subjects
SKIP_CONSTANT = True
TEST_ONLY = False        # if True: skip training and run evaluation using CHECKPOINT_PATH
CHECKPOINT_PATH = None   # e.g. "models_combined/best_somefold_acc_0.9000.pth"
# optional: evaluate only on these folders (list of folder names or filepaths)
TEST_FOLDERS = None

print("Device:", DEVICE)

# ---------- helper: compute normalization stats ----------
def compute_norm_stats(file_list, skip_constant=True):
    psd_acc = []
    plv_acc = []
    for f in file_list:
        try:
            psd, plv, _, _ = load_feature_file(f)
        except Exception:
            continue
        if skip_constant:
            if np.nanmax(psd) - np.nanmin(psd) == 0 or np.nanmax(plv) - np.nanmin(plv) == 0:
                continue
        psd = np.array(psd).reshape(-1)
        plv = np.array(plv).reshape(-1)
        psd_acc.append(np.log1p(np.abs(psd)))
        plv_acc.append(plv)
    if len(psd_acc) == 0:
        raise RuntimeError("No valid files to compute norm stats")
    psd_all = np.concatenate(psd_acc)
    plv_all = np.concatenate(plv_acc)
    return float(psd_all.mean()), float(psd_all.std()), float(plv_all.mean()), float(plv_all.std())

def make_transform(psd_mean, psd_std, plv_mean, plv_std):
    eps = 1e-12
    def transform(psd, plv):
        psd = np.log1p(np.abs(psd))
        if psd_std is None or psd_std == 0:
            psd_norm = (psd - (psd_mean or 0.0))
        else:
            psd_norm = (psd - psd_mean) / (psd_std + eps)
        if plv_std is None or plv_std == 0:
            plv_norm = (plv - (plv_mean or 0.0))
        else:
            plv_norm = (plv - plv_mean) / (plv_std + eps)
        return psd_norm.astype(np.float32), plv_norm.astype(np.float32)
    return transform

# ---------- prepare file list and norms ----------
all_files = []
for fld in FEATURE_FOLDERS:
    all_files += glob.glob(os.path.join(fld, "*_features.npz"))
all_files = sorted(all_files)
print("Total feature files found across folders:", len(all_files))
psd_mean = psd_std = plv_mean = plv_std = None
try:
    psd_mean, psd_std, plv_mean, plv_std = compute_norm_stats(all_files, skip_constant=SKIP_CONSTANT)
    print("norm stats psd_mean,psd_std, plv_mean,plv_std:", psd_mean, psd_std, plv_mean, plv_std)
except Exception as e:
    print("Could not compute norm stats:", e)

transform = make_transform(psd_mean, psd_std, plv_mean, plv_std)

# ---------- build dataset ----------
dataset = EEGFeatureDataset(FEATURE_FOLDERS, transform=transform, skip_constant=SKIP_CONSTANT)
print("Total samples:", len(dataset), "Class counts:", dataset.class_counts())

# get dims
psd0, plv0, lbl0, subj0 = dataset[0]
psd_dim = psd0.size
plv_dim = plv0.size
print("psd_dim", psd_dim, "plv_dim", plv_dim)

# class-weighted loss
counts = dataset.class_counts()
cnt0 = counts.get(0,1)
cnt1 = counts.get(1,1)
w0 = 1.0 / max(1, cnt0)
w1 = 1.0 / max(1, cnt1)
weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
print("Using class weights (0,1):", w0, w1)
criterion = nn.CrossEntropyLoss(weight=weights)

# ---------- training / evaluation helpers ----------
def train_fold(train_idx, val_idx, fold_name="fold"):
    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)

    model = MCNet(psd_dim, plv_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        t0 = time.time()
        for batch in train_loader:
            # dataset returns (psd,plv,label,subj)
            psd = batch[0].view(batch[0].size(0), -1).to(DEVICE).float()
            plv = batch[1].view(batch[1].size(0), -1).to(DEVICE).float()
            labels = batch[2].to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(psd, plv)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # validate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                psd = batch[0].view(batch[0].size(0), -1).to(DEVICE).float()
                plv = batch[1].view(batch[1].size(0), -1).to(DEVICE).float()
                labels = batch[2].to(DEVICE).long()

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
                "val_f1": f1,
                "psd_mean": psd_mean, "psd_std": psd_std, "plv_mean": plv_mean, "plv_std": plv_std
            }
            fname = os.path.join(SAVE_DIR, f"best_{fold_name}_acc_{acc:.4f}.pth")
            torch.save(best_state, fname)
    return best_val_acc

def evaluate_checkpoint(checkpoint_path, test_idx=None):
    ck = torch.load(checkpoint_path, map_location=DEVICE)
    model = MCNet(psd_dim, plv_dim).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    # build loader
    if test_idx is None:
        test_indices = list(range(len(dataset)))
    else:
        test_indices = test_idx
    test_sub = Subset(dataset, test_indices)
    test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            psd = batch[0].view(batch[0].size(0), -1).to(DEVICE).float()
            plv = batch[1].view(batch[1].size(0), -1).to(DEVICE).float()
            labels = batch[2].to(DEVICE).long()
            outputs = model(psd, plv)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true)>0 else 0.0
    cm = confusion_matrix(y_true, y_pred) if len(y_true)>0 else None
    print("Eval checkpoint:", checkpoint_path)
    print("  samples:", len(y_true))
    print("  acc:", acc, "f1:", f1)
    print("  confusion_matrix:\n", cm)
    return acc, f1, cm

# ---------- RUN ----------
if TEST_ONLY:
    if CHECKPOINT_PATH is None:
        raise RuntimeError("Set CHECKPOINT_PATH for TEST_ONLY mode")
    # optional: test only on files from TEST_FOLDERS
    if TEST_FOLDERS:
        all_test_files = []
        for fld in TEST_FOLDERS:
            all_test_files += glob.glob(os.path.join(fld, "*_features.npz"))
        all_test_files = set(all_test_files)
        test_idx = [i for i,(p,_,_) in enumerate(dataset.get_items()) if p in all_test_files]
    else:
        test_idx = None
    evaluate_checkpoint(CHECKPOINT_PATH, test_idx=test_idx)
    raise SystemExit(0)

overall_best = 0.0
if USE_LOSO:
    subj_map = dataset.subjects_map()
    usable_subjs = sorted([s for s in subj_map.keys() if len(subj_map[s]) >= 2])
    print("Detected subjects:", len(subj_map), "Usable for LOSO:", len(usable_subjs))
    fold = 0
    for test_sub in usable_subjs:
        fold += 1
        all_indices = list(range(len(dataset.get_items())))
        test_files = set(path for path,_ in subj_map[test_sub])
        test_idx = [i for i,(path,_,_) in enumerate(dataset.get_items()) if path in test_files]
        train_idx = [i for i in all_indices if i not in test_idx]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        print(f"----- LOSO fold {fold}/{len(usable_subjs)} test subject: {test_sub} -----")
        best_val = train_fold(train_idx, test_idx, fold_name=test_sub)
        print("Fold finished. Best val acc:", best_val)
        overall_best = max(overall_best, best_val)
    print("ALL LOSO finished. Best val acc overall:", overall_best)
else:
    n = len(dataset)
    n_val = max(1, int(n * VAL_RATIO))
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    print("Random-split train/val sizes:", len(train_idx), len(val_idx))
    best_val = train_fold(train_idx, val_idx, fold_name="randomsplit")
    print("Done. Best val acc:", best_val)
    overall_best = max(overall_best, best_val)

print("Training complete. Best overall val acc:", overall_best)

