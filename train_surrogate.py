"""
Surrogate Training
==================
Trains CCUSurrogate (6 inputs → 3 outputs) on the merged dataset.
Saves model.pt and scalers.pkl to models/surrogate/.

Usage
-----
    python train_surrogate.py --data data/ccu_merged.csv
    python train_surrogate.py --data data/ccu_merged.csv --width 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from src.surrogate import CCUSurrogate, X_COLS, Y_COLS

OUT_DIR = Path("models/surrogate")


def load(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "valid" in df.columns:
            df = df[df["valid"] == True]
        frames.append(df)
        print(f"  {Path(p).name}: {len(df):,} rows")
    merged = pd.concat(frames).drop_duplicates(X_COLS).dropna(subset=X_COLS + Y_COLS)
    print(f"  Total: {len(merged):,} points\n")
    return (merged[X_COLS].values.astype("float32"),
            merged[Y_COLS].values.astype("float32"))


def make_loaders(X, y, batch, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    nv = int(len(X) * 0.15)
    nt = int(len(X) * 0.15)
    tr = idx[:len(X)-nv-nt]
    va = idx[len(X)-nv-nt:len(X)-nt]
    te = idx[len(X)-nt:]

    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xtr = torch.tensor(sx.fit_transform(X[tr]))
    ytr = torch.tensor(sy.fit_transform(y[tr]))
    Xva = torch.tensor(sx.transform(X[va]))
    yva = torch.tensor(sy.transform(y[va]))
    Xte = torch.tensor(sx.transform(X[te]))
    yte = torch.tensor(sy.transform(y[te]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((sx, sy), OUT_DIR / "scalers.pkl")
    print(f"  Scalers saved → {OUT_DIR / 'scalers.pkl'}")
    print(f"  Train={len(tr):,}  Val={len(va):,}  Test={len(te):,}")

    def mkdl(Xa, ya, sh):
        return DataLoader(TensorDataset(Xa, ya), batch_size=batch, shuffle=sh)
    return mkdl(Xtr, ytr, True), mkdl(Xva, yva, False), mkdl(Xte, yte, False), sy


def train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 55)
    print("SURROGATE TRAINING")
    print("=" * 55)
    print(f"  Inputs : {X_COLS}")
    print(f"  Outputs: {Y_COLS}\n")

    X, y = load(args.data)
    train_dl, val_dl, test_dl, sy = make_loaders(X, y, args.batch)

    model   = CCUSurrogate(width=args.width)
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  opt, patience=25, factor=0.5, min_lr=1e-6)

    print(f"\n  Width={args.width}  Params={model.n_params:,}\n")

    best_val, patience_cnt, best_state = float("inf"), 0, None
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")

    for epoch in pbar:
        model.train()
        tl_list = []
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl_list.append(loss.item())
        tl = float(np.mean(tl_list))

        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(model(xb), yb).item()
                                 for xb, yb in val_dl]))
        sched.step(vl)
        pbar.set_postfix(train=f"{tl:.5f}", val=f"{vl:.5f}",
                         lr=f"{opt.param_groups[0]['lr']:.1e}",
                         best=f"{best_val:.5f}")

        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                tqdm.write(f"\n  Early stop at epoch {epoch}  best_val={best_val:.6f}")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "width": args.width},
               OUT_DIR / "model.pt")
    print(f"\n  Model saved → {OUT_DIR / 'model.pt'}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    yp = sy.inverse_transform(np.vstack(preds))
    yt = sy.inverse_transform(np.vstack(trues))

    print("\n  Test set results:")
    all_ok = True
    for i, col in enumerate(Y_COLS):
        r2  = r2_score(yt[:, i], yp[:, i])
        mae = mean_absolute_error(yt[:, i], yp[:, i])
        ok  = r2 >= 0.99
        print(f"  {col:<20}  R²={r2:.4f}  MAE={mae:.4f}  "
              f"{'✓' if ok else '✗  → try --width 128'}")
        if not ok:
            all_ok = False
    if all_ok:
        print("\n  All R² ≥ 0.99 — surrogate ready.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     nargs="+", required=True)
    p.add_argument("--epochs",   type=int,   default=500)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--width",    type=int,   default=64)
    p.add_argument("--patience", type=int,   default=60)
    train(p.parse_args())


if __name__ == "__main__":
    main()
