# train_mcnet_iowa.py
import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from dataset_loader import EEGFeatureDataset, load_feature_file
from collections import Counter

# ---------------- CONFIG ----------------
FEATURE_FOLDERS = ["features/Iowa"]
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
VAL_RATIO = 0.2
SAVE_DIR = "models_iowa"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LOSO = True   # True: LOSO; False: random train/val split (quicker)
SKIP_CONSTANT = True

print("Device:", DEVICE)

# ---------------- helper: compute normalization stats ----------------
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
        # use log1p(abs) for PSD to stabilize tiny numbers
        psd_acc.append(np.log1p(np.abs(psd)))
        plv_acc.append(plv)
    if len(psd_acc) == 0:
        raise RuntimeError("No valid files to compute norm stats")
    psd_all = np.concatenate(psd_acc)
    plv_all = np.concatenate(plv_acc)
    return float(psd_all.mean()), float(psd_all.std()), float(plv_all.mean()), float(plv_all.std())

# ---------------- MODEL ----------------
class MCNet(nn.Module):
    def __init__(self, psd_dim, plv_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.psd_net = nn.Sequential(
            nn.Linear(psd_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.ReLU()
        )
        self.plv_net = nn.Sequential(
            nn.Linear(plv_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    def forward(self, psd, plv):
        x1 = self.psd_net(psd)
        x2 = self.plv_net(plv)
        x = torch.cat([x1,x2], dim=1)
        out = self.classifier(x)
        return out

# ---------------- prepare file list and norm stats ----------------
all_files = sorted(glob.glob(os.path.join(FEATURE_FOLDERS[0], "*_features.npz")))
print("Total feature files found:", len(all_files))
psd_mean = psd_std = plv_mean = plv_std = None
try:
    psd_mean, psd_std, plv_mean, plv_std = compute_norm_stats(all_files, skip_constant=SKIP_CONSTANT)
    print("norm stats psd_mean,psd_std, plv_mean,plv_std:", psd_mean, psd_std, plv_mean, plv_std)
except Exception as e:
    print("Could not compute norm stats:", e)

# define transform closure
def make_transform(psd_mean, psd_std, plv_mean, plv_std):
    eps = 1e-12
    def transform(psd, plv):
        # psd: numpy 1d
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

# ---------------- dataset ----------------
dataset = EEGFeatureDataset(FEATURE_FOLDERS, transform=transform, skip_constant=SKIP_CONSTANT)
print("Total samples:", len(dataset), "Class counts:", dataset.class_counts())
psd0, plv0, _ = dataset[0]
psd_dim = psd0.size
plv_dim = plv0.size
print("psd_dim", psd_dim, "plv_dim", plv_dim)

# class weights for imbalanced data
counts = dataset.class_counts()
cnt0 = counts.get(0,1)
cnt1 = counts.get(1,1)
w0 = 1.0 / max(1, cnt0)
w1 = 1.0 / max(1, cnt1)
weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
print("Using class weights (0,1):", w0, w1)

criterion = nn.CrossEntropyLoss(weight=weights)

# ---------------- training routine ----------------
def run_train_val(train_idx, val_idx, run_id="run"):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
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
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for psd, plv, labels in val_loader:
                psd = psd.view(psd.size(0), -1).to(DEVICE)
                plv = plv.view(plv.size(0), -1).to(DEVICE)
                outputs = model(psd, plv)
                preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.numpy().tolist())
        acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true)>0 else 0.0
        print(f"{run_id} epoch {epoch}/{EPOCHS} loss={np.mean(losses):.4f} val_acc={acc:.4f} val_f1={f1:.4f} time={(time.time()-t0):.1f}s")
        if acc > best_val:
            best_val = acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": acc
            }, os.path.join(SAVE_DIR, f"best_{run_id}_acc_{acc:.4f}.pth"))
    return best_val

# ---------------- run ----------------
if USE_LOSO:
    subj_map = dataset.subjects_map()
    usable_subjs = sorted([s for s in subj_map.keys() if len(subj_map[s]) >= 2])
    print("Detected subjects:", len(subj_map), "Usable for LOSO:", len(usable_subjs))
    overall_best = 0.0
    fold = 0
    for test_sub in usable_subjs:
        fold += 1
        all_indices = list(range(len(dataset.items)))
        test_files = set(path for path,_ in subj_map[test_sub])
        test_idx = [i for i,(path,_) in enumerate(dataset.items) if path in test_files]
        train_idx = [i for i in all_indices if i not in test_idx]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        print(f"----- LOSO fold {fold}/{len(usable_subjs)} test subject: {test_sub} -----")
        best_val = run_train_val(train_idx, test_idx, run_id=test_sub)
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
    best_val = run_train_val(train_idx, val_idx, run_id="randomsplit")
    print("Done. Best val acc:", best_val)

print("Training complete.")
