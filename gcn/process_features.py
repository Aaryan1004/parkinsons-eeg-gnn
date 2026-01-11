import os, warnings, pickle, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from scipy.signal import welch, butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F
import mne
import networkx as nx

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "processed"        # where cleaned EDFs are saved
OUTPUT_DIR = "features"       # where outputs will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SEC = 1.0
BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,12), 'beta': (13,30), 'gamma': (30,40)}
PLV_BAND = (13,30)
K = 6   # k for mutual knn graph

# ----------------------------
# Helper functions
# ----------------------------
def compute_node_feats(epoch, sf, n_ch):
    abs_p = np.zeros((n_ch, len(BANDS)))
    rel_p = np.zeros((n_ch, len(BANDS)))
    for ch in range(n_ch):
        f, Pxx = welch(epoch[ch], fs=sf, nperseg=max(256, sf*2), noverlap=int(0.5*sf))
        total_idx = (f>=0.5)&(f<=40)
        total_power = np.trapz(Pxx[total_idx], f[total_idx]) + 1e-12
        for bi,(bname,(bmin,bmax)) in enumerate(BANDS.items()):
            idx = (f>=bmin)&(f<=bmax)
            bp = np.trapz(Pxx[idx], f[idx])
            abs_p[ch,bi] = bp
            rel_p[ch,bi] = bp/total_power
    return np.hstack([abs_p, rel_p])  # (n_ch, 2*len(BANDS))

def compute_plv_mat(epoch, fmin, fmax, sf, n_ch):
    b,a = butter(4, [fmin/(sf/2), fmax/(sf/2)], btype='band')
    filt = filtfilt(b, a, epoch, axis=1)
    phases = np.angle(hilbert(filt, axis=1))
    mat = np.zeros((n_ch, n_ch))
    for i,j in combinations(range(n_ch),2):
        dphi = phases[i] - phases[j]
        mat[i,j] = mat[j,i] = np.abs(np.mean(np.exp(1j*dphi)))
    return mat

def mutual_knn(A, k=6):
    N = A.shape[0]
    edges = set()
    for i in range(N):
        neigh = np.argsort(A[i])[::-1]
        neigh = [x for x in neigh if x!=i][:k]
        for j in neigh:
            edges.add((i,j))
    mutual = [(i,j) for (i,j) in edges if (j,i) in edges]
    if len(mutual)==0:
        mutual = []
        for i in range(N):
            neigh = np.argsort(A[i])[::-1][:k]
            for j in neigh:
                if i<j:
                    mutual.append((i,j))
    A_sparse = np.zeros_like(A)
    for (i,j) in mutual:
        A_sparse[i,j] = A[i,j]; A_sparse[j,i] = A[j,i]
    if A_sparse.sum()==0:
        return A
    return A_sparse

def normalize_adj(A):
    A = A + np.eye(A.shape[0])
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5); deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D = np.diag(deg_inv_sqrt)
    return D @ A @ D

# ----------------------------
# GCN model
# ----------------------------
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * math.sqrt(2.0/(in_dim+out_dim)))
        self.b = nn.Parameter(torch.zeros(out_dim))
    def forward(self, X, A_norm):
        return A_norm @ (X @ self.W) + self.b

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=2):
        super().__init__()
        self.g1 = DenseGCNLayer(in_dim, hidden)
        self.g2 = DenseGCNLayer(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)
    def forward(self, X, A):
        A_norm = torch.tensor(normalize_adj(A), dtype=torch.float32, device=device)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        h = F.relu(self.g1(X, A_norm))
        h = F.relu(self.g2(h, A_norm))
        g = h.mean(dim=0, keepdim=True)  # graph pooling
        out = self.lin(g)
        return out.squeeze(0)

# ----------------------------
# Main pipeline for one EDF
# ----------------------------
def process_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sf = int(raw.info['sfreq'])
    data = raw.get_data()
    n_ch, n_times = data.shape
    win_samps = int(WINDOW_SEC * sf)
    n_epochs = n_times // win_samps
    print(f"\nEDF {edf_path}: {n_ch} channels, {sf} Hz, {n_epochs} epochs")

    node_features_all = []
    plv_mats_all = []
    graphs = []

    for e in range(n_epochs):
        start = e * win_samps
        epoch = data[:, start:start+win_samps]
        nf = compute_node_feats(epoch, sf, n_ch)
        plv = compute_plv_mat(epoch, PLV_BAND[0], PLV_BAND[1], sf, n_ch)
        A_sparse = mutual_knn(plv, k=K)
        node_features_all.append(nf)
        plv_mats_all.append(plv)
        graphs.append({'x': nf.astype(np.float32), 'A': A_sparse.astype(np.float32)})

    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    np.save(os.path.join(OUTPUT_DIR, f"{base_name}_node_feats.npy"), np.array(node_features_all))
    np.save(os.path.join(OUTPUT_DIR, f"{base_name}_plv_mats.npy"), np.array(plv_mats_all))
    with open(os.path.join(OUTPUT_DIR, f"{base_name}_graphs_epoch.pkl"), "wb") as f:
        pickle.dump(graphs, f)

    # Export CSV for epoch 0 and mean across epochs
    band_names = list(BANDS.keys())
    cols = [f"{b}_abs" for b in band_names] + [f"{b}_rel" for b in band_names]

    df_feats = pd.DataFrame(node_features_all[0], columns=cols)
    df_feats.to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_node_features_epoch0.csv"), index=False)

    df_plv = pd.DataFrame(plv_mats_all[0])
    df_plv.to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_plv_matrix_epoch0.csv"), index=False)

    # FIXED: keep per-channel means (shape = n_ch × n_features)
    df_feats_mean = pd.DataFrame(np.mean(node_features_all, axis=0), columns=cols)
    df_feats_mean.to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_node_features_mean.csv"), index=False)

    # Quick plots
    plt.figure(figsize=(8,6))
    sns.heatmap(df_feats.T, cmap="viridis", cbar_kws={'label':'Power'}, xticklabels=False)
    plt.ylabel("Band"); plt.xlabel("Channel index")
    plt.title(f"{base_name} — Epoch 0 Band Power")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_bandpower_epoch0.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(plv_mats_all[0], cmap="magma", vmin=0, vmax=1)
    plt.title(f"{base_name} — Epoch 0 PLV")
    plt.xlabel("Channel"); plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_plv_epoch0.png"), dpi=300)
    plt.close()

    # NetworkX graph visualization for epoch 0
    A = graphs[0]['A']
    G = nx.Graph()
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[0]):
            if A[i,j] > 0:
                G.add_edge(i, j, weight=A[i,j])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_nodes(G, pos, node_size=80)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(f"{base_name} — Epoch 0 Graph")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_graph_epoch0.png"), dpi=300)
    plt.close()

    print(f"✅ Saved features & plots for {base_name} in {OUTPUT_DIR}")
    return graphs, np.array(node_features_all), np.array(plv_mats_all)

# ----------------------------
# Run on all EDFs
# ----------------------------
if __name__ == "__main__":
    edf_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_clean.edf")]
    if len(edf_files) == 0:
        print("No EDF files found in 'processed'. Run preprocess.py first!")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_graphs = []

        for file in edf_files:
            edf_path = os.path.join(DATA_DIR, file)
            graphs, node_feats, plv_mats = process_edf(edf_path)
            all_graphs.extend(graphs)

        # ----------------------------
        # Toy GCN training (dummy labels)
        # ----------------------------
        y_epochs = np.ones(len(all_graphs), dtype=int)  # dummy PD labels
        train_idx, test_idx = train_test_split(np.arange(len(all_graphs)), test_size=0.2, random_state=0)

        model = GCNModel(in_dim=all_graphs[0]['x'].shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(10):
            model.train()
            losses = []
            for i in train_idx:
                g = all_graphs[i]
                out = model(g['x'], g['A'])
                y = torch.tensor([y_epochs[i]], dtype=torch.long, device=device)
                loss = F.cross_entropy(out.unsqueeze(0), y)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            print(f"Epoch {epoch}, mean loss {np.mean(losses):.4f}")
