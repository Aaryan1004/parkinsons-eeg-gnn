"""
train.py

Trains a simple dense GCN on your per-epoch graph dataset saved in features/*_graphs_epoch.pkl.
This script:
 - Splits by subject (no epoch-level leakage)
 - Handles variable node-count graphs (uses custom collate)
 - Trains on GPU if available
 - Prints accuracy, AUC, recall and confusion matrix on test set
"""

import os
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# CONFIG
# ----------------------------
FEATURE_DIR = "features"
LABELS_FILE = "labels.csv"
RANDOM_STATE = 42

BATCH_SIZE = 8          # number of graphs per batch (each graph may have different node counts)
LR = 1e-3
EPOCHS = 12
HIDDEN = 64
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SEC = 1.0  # unused here, but kept for context

# ----------------------------
# Helper: load subject -> label mapping, and available subjects
# ----------------------------
def get_subject_list_and_labels(labels_csv=LABELS_FILE, feature_dir=FEATURE_DIR):
    df = pd.read_csv(labels_csv)
    subjects = []
    labels = []
    for _, row in df.iterrows():
        subj = row["subject"]
        pkl_path = os.path.join(feature_dir, f"{subj}_graphs_epoch.pkl")
        if os.path.exists(pkl_path):
            subjects.append(subj)
            labels.append(int(row["label"]))
        else:
            print(f"⚠️ Warning: missing {pkl_path} — skipping subject {subj}")
    return subjects, labels

# ----------------------------
# Dataset that takes a list of subjects and returns all their graphs (per-epoch)
# ----------------------------
class SubjectGraphsDataset(Dataset):
    def __init__(self, subjects, labels_map, feature_dir=FEATURE_DIR, transform=None):
        """
        subjects: list of subject ids (strings)
        labels_map: dict subject -> label
        """
        self.feature_dir = feature_dir
        self.transform = transform
        self.samples = []  # list of (graph_dict, label)

        for subj in subjects:
            pkl_path = os.path.join(feature_dir, f"{subj}_graphs_epoch.pkl")
            with open(pkl_path, "rb") as f:
                graphs = pickle.load(f)
            lbl = int(labels_map[subj])
            # append each epoch-graph with its subject label
            for g in graphs:
                self.samples.append((g, lbl))

        print(f"Loaded {len(self.samples)} graphs from {len(subjects)} subjects.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g, label = self.samples[idx]
        x = torch.tensor(g["x"], dtype=torch.float32)      # (n_nodes, n_features)
        A = torch.tensor(g["A"], dtype=torch.float32)      # (n_nodes, n_nodes)
        y = torch.tensor(label, dtype=torch.long)
        if self.transform:
            x, A = self.transform(x, A)
        return x, A, y

# ----------------------------
# Custom collate for variable-size graphs
# ----------------------------
def custom_collate(batch):
    xs, As, ys = zip(*batch)
    return list(xs), list(As), torch.tensor(ys, dtype=torch.long)

# ----------------------------
# GCN model (dense adjacency)
# ----------------------------
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * math.sqrt(2.0/(in_dim+out_dim)))
        self.b = nn.Parameter(torch.zeros(out_dim))
    def forward(self, X, A_norm):
        # X: (n_nodes, in_dim), A_norm: (n_nodes, n_nodes)
        return A_norm @ (X @ self.W) + self.b

def normalize_adj_torch(A):
    # A: torch.tensor (n_nodes, n_nodes)
    device = A.device
    I = torch.eye(A.size(0), device=device)
    A_hat = A + I
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return D @ A_hat @ D

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.g1 = DenseGCNLayer(in_dim, hidden)
        self.g2 = DenseGCNLayer(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)
    def forward(self, X, A):
        # X, A expected as torch tensors on the correct device
        A_norm = normalize_adj_torch(A)
        h = F.relu(self.g1(X, A_norm))
        h = F.relu(self.g2(h, A_norm))
        g = h.mean(dim=0, keepdim=True)   # (1, hidden) global average pooling
        out = self.lin(g)                 # (1, num_classes)
        return out.squeeze(0)             # (num_classes,)

# ----------------------------
# Training + Eval helpers
# ----------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for xs, As, ys in loader:
        optimizer.zero_grad()
        batch_loss = 0.0
        # xs, As are lists; ys is tensor length=batch_size
        for i in range(len(xs)):
            x = xs[i].to(device)   # (n_nodes, n_features)
            A = As[i].to(device)   # (n_nodes, n_nodes)
            y = ys[i].to(device)   # scalar
            out = model(x, A)      # (num_classes,)
            loss = F.cross_entropy(out.unsqueeze(0), y.unsqueeze(0))
            batch_loss = batch_loss + loss
        batch_loss = batch_loss / len(xs)  # average per-graph loss in this batch
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds = []
    probs = []
    trues = []
    with torch.no_grad():
        for xs, As, ys in loader:
            for i in range(len(xs)):
                x = xs[i].to(device)
                A = As[i].to(device)
                y = ys[i].item()
                out = model(x, A)            # (num_classes,)
                p = F.softmax(out, dim=0).cpu().numpy()
                pred = int(out.argmax().cpu().item())
                preds.append(pred)
                probs.append(p[1])          # probability for class 1 (PD)
                trues.append(y)
    # compute metrics
    acc = accuracy_score(trues, preds)
    rec = recall_score(trues, preds, zero_division=0)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(trues, preds)
    return {"accuracy": acc, "recall": rec, "auc": auc, "confusion_matrix": cm}

# ----------------------------
# Build subject splits and dataloaders
# ----------------------------
def make_dataloaders(test_size=0.2):
    subjects, labels = get_subject_list_and_labels(LABELS_FILE, FEATURE_DIR)
    labels_map = {s: lab for s, lab in zip(subjects, labels)}
    # stratified split by subject labels
    subj_train, subj_test = train_test_split(subjects, test_size=test_size, random_state=RANDOM_STATE,
                                             stratify=[labels_map[s] for s in subjects])
    print(f"Subjects: {len(subjects)} total, {len(subj_train)} train, {len(subj_test)} test")

    train_dataset = SubjectGraphsDataset(subj_train, labels_map, feature_dir=FEATURE_DIR)
    test_dataset = SubjectGraphsDataset(subj_test, labels_map, feature_dir=FEATURE_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    return train_loader, test_loader

# ----------------------------
# Main training procedure
# ----------------------------
def main():
    print("Device:", DEVICE)
    train_loader, test_loader = make_dataloaders(test_size=0.2)

    # infer in_dim from a sample (first sample in train loader)
    # get one batch
    for xs, As, ys in train_loader:
        sample_x = xs[0]
        in_dim = sample_x.shape[1]
        break

    model = GCNModel(in_dim=in_dim, hidden=HIDDEN, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        metrics = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {loss:.4f}  test_acc: {metrics['accuracy']:.4f}  recall: {metrics['recall']:.4f}  auc: {metrics['auc']:.4f}")
        print(" Confusion matrix:\n", metrics["confusion_matrix"])

        # save best model by AUC
        if not np.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), "best_gcn_model.pth")
            print(f"Saved best model (AUC={best_auc:.4f}) -> best_gcn_model.pth")

    print("Training complete. Best AUC:", best_auc)

if __name__ == "__main__":
    main()
