import os, pickle, numpy as np
import pandas as pd

FEATURE_DIR = "features"

# list all processed subjects
files = [f for f in os.listdir(FEATURE_DIR) if f.endswith("_node_feats.npy")]
print(f"Found {len(files)} subjects.")

for f in files:
    subject = f.replace("_node_feats.npy", "")
    node_feats = np.load(os.path.join(FEATURE_DIR, f))
    plv_mats = np.load(os.path.join(FEATURE_DIR, f"{subject}_plv_mats.npy"))
    with open(os.path.join(FEATURE_DIR, f"{subject}_graphs_epoch.pkl"), "rb") as gfile:
        graphs = pickle.load(gfile)

    print(f"\n--- {subject} ---")
    print("Node feats shape:", node_feats.shape)  # (n_epochs, n_channels, n_features)
    print("PLV mats shape:  ", plv_mats.shape)    # (n_epochs, n_channels, n_channels)
    print("Graphs:", len(graphs))

    # sanity check first graph
    g0 = graphs[0]
    print("Graph 0 -> x:", g0['x'].shape, "A:", g0['A'].shape)

    # preview CSVs
    epoch0_csv = os.path.join(FEATURE_DIR, f"{subject}_node_features_epoch0.csv")
    mean_csv = os.path.join(FEATURE_DIR, f"{subject}_node_features_mean.csv")
    if os.path.exists(epoch0_csv):
        df0 = pd.read_csv(epoch0_csv)
        print("Epoch0 CSV preview:\n", df0.head())
    if os.path.exists(mean_csv):
        dfm = pd.read_csv(mean_csv)
        print("Mean features CSV preview:\n", dfm.head())
