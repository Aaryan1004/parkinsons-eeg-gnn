  # mcnet_model.py
"""
MCNet model file (drop-in for your training scripts).

Usage:
    from mcnet_model import MCNet, init_model, save_model, load_model

The model forward() returns raw logits (no softmax) — suitable with nn.CrossEntropyLoss.
"""

import torch
import torch.nn as nn
import os

class MCNet(nn.Module):
    def __init__(self, psd_dim, plv_dim, hidden=128, dropout=0.3):
        """
        psd_dim : int  -> flattened length of PSD vector (e.g. 32*4 = 128)
        plv_dim : int  -> flattened length of PLV vector (e.g. 496)
        hidden  : int  -> hidden size for each branch
        dropout : float
        """
        super().__init__()
        # small MLP for PSD
        self.psd_net = nn.Sequential(
            nn.Linear(psd_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.ReLU(inplace=True)
        )
        # small MLP for PLV
        self.plv_net = nn.Sequential(
            nn.Linear(plv_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.ReLU(inplace=True)
        )
        # classifier on concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)   # 2 classes: HC / PD
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier init for linear layers, zero bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, psd, plv):
        """
        psd: Tensor shape (B, psd_dim)
        plv: Tensor shape (B, plv_dim)
        returns logits: Tensor shape (B, 2)
        """
        x1 = self.psd_net(psd)
        x2 = self.plv_net(plv)
        x = torch.cat([x1, x2], dim=1)
        out = self.classifier(x)
        return out

    @torch.no_grad()
    def predict(self, psd, plv, device=None):
        """
        Convenience prediction helper that returns class indices.
        Input can be numpy arrays or torch tensors.
        """
        if device is None:
            device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")
        # convert to tensor if needed
        if not isinstance(psd, torch.Tensor):
            psd = torch.tensor(psd, dtype=torch.float32, device=device)
        if not isinstance(plv, torch.Tensor):
            plv = torch.tensor(plv, dtype=torch.float32, device=device)
        if psd.dim() == 1:
            psd = psd.unsqueeze(0)
        if plv.dim() == 1:
            plv = plv.unsqueeze(0)
        logits = self.forward(psd.to(device), plv.to(device))
        preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

# helper factory
def init_model(psd_dim, plv_dim, hidden=128, dropout=0.3, device=None):
    m = MCNet(psd_dim, plv_dim, hidden=hidden, dropout=dropout)
    if device:
        m = m.to(device)
    return m

# save/load helpers (saves state_dict + metadata)
def save_model(model, path, optimizer_state=None, epoch=None, extra=None):
    state = {
        "model_state": model.state_dict()
    }
    if optimizer_state is not None:
        state["optimizer_state"] = optimizer_state
    if epoch is not None:
        state["epoch"] = epoch
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)

def load_model(model, path, map_location=None):
    """
    Load checkpoint dict into provided model instance.
    Returns the loaded checkpoint dict for optimizer/state info if present.
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    return ckpt

# quick smoke test when run standalone
if __name__ == "__main__":
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psd_dim = 128
    plv_dim = 496
    model = init_model(psd_dim, plv_dim, device=device)
    # dummy batch
    b = 4
    psd = torch.randn(b, psd_dim, device=device)
    plv = torch.randn(b, plv_dim, device=device)
    logits = model(psd, plv)
    print("MCNet smoke test logits shape:", logits.shape)  # expect (4,2)
