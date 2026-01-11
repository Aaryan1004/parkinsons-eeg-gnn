# train_mcnet_combined.py
"""
Train MCNet on multiple feature folders (IOWA + UC + UNM ...).
Features: expects "*_features.npz" files as used by your dataset_loader.EEGFeatureDataset.

Behaviors:
 - Computes global normalization stats
 - Uses LOSO by subject (subject id prefixed with folder name to avoid collisions)
 - Uses WeightedRandomSampler on training set to balance classes per fold
 - Saves best model per fold
"""

import os
import glob
import time
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset_loader import EEGFeatureDataset, load_feature_file
from mcnet_model import MCNet, save_model

# ========== CONFIG ==========
FEATURE_FOLDERS = ["features/Iowa", "features/UC"]   # add other folders here
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
SAVE_DIR = "models_combined"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LOSO = True     # LOSO across subjects (recommended for subject-generalization)
SKIP_CONSTANT = True
PRINT_EVERY = 1
# ============================

print("Device:", DEVICE)
print("Folders:", FEATURE_FOLDERS)

# ---------- helper: compute norm stats ----------
def compute_norm_stats_for_folders(folders, skip_constant=True):
    files = []
    for f in folders:
        files += glob.glob(os.path.join(f, "*_features.npz"))
    psd_acc = []
    plv_acc = []
    for p in files:
        try:
            psd, plv, _, _ = load_feature_file(p)
        except Exception:
            continue
        # skip constant if requested
        if skip_constant:
            if np.nanmax(psd) - np.nanmin(psd) == 0 or np.nanmax(plv) - np.nanmin(plv) == 0:
                continue
        psd = np.array(psd).reshape(-1)
        plv = np.array(plv).reshape(-1)
        psd_acc.append(np.log1p(np.abs(psd)))
        plv_acc.append(plv)
    if len(psd_acc) == 0:
        raise RuntimeError("No valid files for norm stats")
    psd_all = np.concatenate(psd_acc)
    plv_all = np.concatenate(plv_acc)
    return float(psd_all.mean()), float(psd_all.std()), float(plv_all.mean()), float(plv_all.std())

print("Computing normalization stats (may take a moment)...")
psd_mean, psd_std, plv_mean, plv_std = compute_norm_stats_for_folders(FEATURE_FOLDERS, skip_constant=SKIP_CONSTANT)
print("norm stats psd_mean,psd_std, plv_mean,plv_std:", psd_mean, psd_std, plv_mean, plv_std)

# ---------- transform closure ----------
eps = 1e-12
def make_transform(psd_mean, psd_std, plv_mean, plv_std):
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

transform = make_transform(psd_mean, psd_std, plv_mean, plv_std)

# ---------- dataset ----------
dataset = EEGFeatureDataset(FEATURE_FOLDERS, transform=transform, skip_constant=SKIP_CONSTANT)
print("Total samples:", len(dataset), "Class counts:", dataset.class_counts())

# get dims from first example
psd0, plv0, _ = dataset[0]
psd_dim = psd0.size
plv_dim = plv0.size
print("psd_dim", psd_dim, "plv_dim", plv_dim)

# compute class weight for CrossEntropy (inverse frequency scaled)
counts = dataset.class_counts()
n_total = counts.get(0,0) + counts.get(1,0)
# avoid zero division
w0 = n_total / (2.0 * max(1, counts.get(0,1)))
w1 = n_total / (2.0 * max(1, counts.get(1,1)))
class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
print("Class weights (for CrossEntropy):", w0, w1)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ---------- helper: build subject map (unique across folders) ----------
# We prefix subject id with the parent folder name to avoid collisions across datasets
def build_subject_map_with_prefix(dataset):
    mp = defaultdict(list)
    for idx, (path, label) in enumerate(dataset.items):
        # subject base 'xxx' extracted like your subjects_map idea
        base = os.path.basename(path).split('_epoch')[0]
        folder_prefix = os.path.basename(os.path.dirname(path))
        subj = f"{folder_prefix}__{base}".lower()
        mp[subj].append((idx, path, label))
    return mp

subj_map = build_subject_map_with_prefix(dataset)
usable_subjs = sorted([s for s,v in subj_map.items() if len(v) >= 2])
print("Detected subjects:", len(subj_map), "Usable for LOSO:", len(usable_subjs))

# ---------- training / evaluation helpers ----------
def evaluate_model_on_loader(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for psd, plv, labels in loader:
            psd = psd.view(psd.size(0), -1).to(DEVICE)
            plv = plv.view(plv.size(0), -1).to(DEVICE)
            out = model(psd, plv)
            preds = torch.argmax(out, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
    if len(y_true) == 0:
        return 0.0, 0.0, None
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return acc, f1, cm

def run_train_fold(train_idx, val_idx, run_id="run"):
    # create samplers to balance classes in train set
    train_labels = [dataset.items[i][1] for i in train_idx]
    label_counts = Counter(train_labels)
    # per-sample weight = 1/count(label)
    sample_weights = [1.0 / label_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = MCNet(psd_dim, plv_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        t0 = time.time()
        for psd, plv, labels in train_loader:
            psd = psd.view(psd.size(0), -1).to(DEVICE)
            plv = plv.view(plv.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(psd, plv)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        acc, f1, cm = evaluate_model_on_loader(model, val_loader)
        if epoch % PRINT_EVERY == 0:
            print(f"{run_id} epoch {epoch}/{EPOCHS} loss={np.mean(losses):.4f} val_acc={acc:.4f} val_f1={f1:.4f} time={(time.time()-t0):.1f}s")
            if cm is not None:
                print("  val_confusion:", cm.tolist())
        if acc > best_val:
            best_val = acc
            save_path = os.path.join(SAVE_DIR, f"best_{run_id}_acc_{acc:.4f}.pth")
            save_model(model, save_path, optimizer_state=optimizer.state_dict(), epoch=epoch, extra={"run_id": run_id})
    return best_val

# ---------- main run ----------
overall_best = 0.0
if USE_LOSO:
    fold = 0
    for subj in usable_subjs:
        fold += 1
        # indices for test subject
        test_idx = [idx for idx,_,_ in subj_map[subj]]
        train_idx = [i for i in range(len(dataset.items)) if i not in test_idx]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        print(f"----- LOSO fold {fold}/{len(usable_subjs)} test subject: {subj} -----")
        best_val = run_train_fold(train_idx, test_idx, run_id=subj)
        print("Fold finished. Best val acc:", best_val)
        overall_best = max(overall_best, best_val)
    print("ALL LOSO finished. Best val acc overall:", overall_best)
else:
    # simple random split
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    best_val = run_train_fold(train_idx, val_idx, run_id="randomsplit")
    print("Done. Best val acc:", best_val)

print("Training complete.")

