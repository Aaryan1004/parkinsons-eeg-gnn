import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# paths
FEATURE_DIR = "features"
LABELS_FILE = "labels.csv"

class EEGGraphDataset(Dataset):
    def __init__(self, labels_csv=LABELS_FILE, feature_dir=FEATURE_DIR, transform=None):
        self.labels_df = pd.read_csv(labels_csv)
        self.feature_dir = feature_dir
        self.transform = transform

        # collect all samples (graphs + labels)
        self.samples = []
        for _, row in self.labels_df.iterrows():
            subj = row["subject"]
            label = int(row["label"])
            pkl_path = os.path.join(feature_dir, f"{subj}_graphs_epoch.pkl")

            if not os.path.exists(pkl_path):
                print(f"⚠️ Warning: missing {pkl_path}, skipping")
                continue

            with open(pkl_path, "rb") as f:
                graphs = pickle.load(f)

            for g in graphs:
                self.samples.append((g, label))

        print(f"✅ Loaded {len(self.samples)} graphs from {len(self.labels_df)} subjects.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g, label = self.samples[idx]
        x = torch.tensor(g["x"], dtype=torch.float32)      # shape (n_nodes, 10)
        A = torch.tensor(g["A"], dtype=torch.float32)      # shape (n_nodes, n_nodes)
        y = torch.tensor(label, dtype=torch.long)

        if self.transform:
            x, A = self.transform(x, A)
        return x, A, y


# ------------------------------
# Custom collate for variable-size graphs
# ------------------------------
def custom_collate(batch):
    xs, As, ys = zip(*batch)  # xs = list of node feats, As = list of adj, ys = tuple of labels
    return list(xs), list(As), torch.tensor(ys)


# quick test
if __name__ == "__main__":
    dataset = EEGGraphDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    for xs, As, ys in loader:
        print(f"Batch size: {len(xs)}")
        for i in range(len(xs)):
            print(f" Graph {i}: x={xs[i].shape}, A={As[i].shape}, y={ys[i].item()}")
        break
