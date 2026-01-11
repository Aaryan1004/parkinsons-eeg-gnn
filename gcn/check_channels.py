import os, pickle

FEATURE_DIR = "features"

for f in os.listdir(FEATURE_DIR):
    if f.endswith("_graphs_epoch.pkl"):
        path = os.path.join(FEATURE_DIR, f)
        with open(path, "rb") as handle:
            graphs = pickle.load(handle)
            g0 = graphs[0]
            print(f"{f}: {g0['x'].shape[0]} channels")
