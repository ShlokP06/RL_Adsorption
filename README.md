# Adsorber — RL Control for MEA CO2 Capture

A reinforcement learning controller for optimizing **post-combustion CO2 capture** using a monoethanolamine (MEA) absorber-stripper system. A PPO-LSTM agent learns to maximize capture rate while minimizing energy consumption, outperforming a classical PID baseline.

---

## Overview

The project replaces a traditional PID controller with a neural RL agent trained via **Proximal Policy Optimization with LSTM** (RecurrentPPO). Rather than running the expensive physics simulation at every timestep, the agent acts on a fast **neural surrogate model** that approximates the absorber-stripper process.

**Control variables (actions):**
| Variable | Range | Description |
|---|---|---|
| `L_liq` | 2–12 kg/m²/s | Solvent flowrate |
| `alpha_lean` | 0.18–0.38 mol/mol | Lean solvent loading |
| `T_L_in` | 30–55 °C | Solvent inlet temperature |
| `T_ic` | 25–50 °C | Mid-column intercooling temperature |

**Process outputs (observations):**
| Variable | Typical range | Description |
|---|---|---|
| `capture_rate` | 75–97 % | CO2 removal efficiency |
| `E_specific` | 3.5–6.5 GJ/t | Regeneration energy per tonne CO2 |
| `alpha_rich` | — mol/mol | Rich solvent loading |

---

## Project Structure

```
Adsorber/
├── src/
│   ├── simulation.py        # Rate-based absorber-stripper physics model
│   ├── surrogate.py         # Neural surrogate (6 inputs → 3 outputs)
│   └── env.py               # Gymnasium RL environment
├── data/
│   └── merged_ccu.csv       # Training dataset (~8.5 MB)
├── models/
│   ├── surrogate/           # Trained surrogate weights + scalers
│   └── rl/best/             # Best PPO-LSTM checkpoint
├── results/                 # Evaluation plots and CSVs
├── logs/                    # TensorBoard training logs
├── generate_data.py         # Latin Hypercube Sampling data generation
├── merge_data.py            # Dataset merging and deduplication
├── train_surrogate.py       # Surrogate model training
├── train_rl.py              # PPO-LSTM agent training
├── compare_controllers.py   # RL vs PID comparison
└── sensitivity_analysis.py  # One-at-a-time sensitivity analysis
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline

### 1. Generate simulation data

Uses Latin Hypercube Sampling over the operating space:

```bash
python generate_data.py --n 10000 --seed 101 --out data/batch1.csv
python generate_data.py --n 10000 --seed 404 --out data/batch2.csv --wide
```

`--wide` expands the sampling bounds for robustness.

### 2. Merge datasets

```bash
python merge_data.py --files "data/batch*.csv" --out data/merged_ccu.csv
```

### 3. Train the surrogate model

Trains a 4-layer feedforward network targeting R² ≥ 0.99 on all outputs:

```bash
python train_surrogate.py --data data/merged_ccu.csv --width 128
```

Outputs: `models/surrogate/model.pt`, `models/surrogate/scalers.pkl`

### 4. Train the RL controller

```bash
python train_rl.py --timesteps 500000 --n-envs 4
```

Training uses **curriculum learning** across 3 phases:
- **Phase 0** — frozen disturbances, random initial conditions
- **Phase 1** — Ornstein-Uhlenbeck drift on gas flowrate and CO2 composition
- **Phase 2** — full dynamics with random step changes

### 5. Evaluate the trained agent

```bash
python train_rl.py --eval-only --model models/rl/best/best_model.zip
```

### 6. Compare RL vs PID

```bash
python compare_controllers.py --scenario 1   # step in G_gas
python compare_controllers.py --scenario 2   # step in y_CO2_in
python compare_controllers.py --scenario 3   # combined disturbance
```

Outputs a dashboard PNG and transient CSV to `results/`.

### 7. Sensitivity analysis

```bash
python sensitivity_analysis.py
```

Runs one-at-a-time (OAT) sweeps across all 6 inputs, producing heatmaps in `results/`.

---

## Architecture

### Surrogate model (`src/surrogate.py`)

| | |
|---|---|
| Inputs | G_gas, L_liq, y_CO2_in, T_L_in, alpha_lean, T_ic |
| Outputs | capture_rate, E_specific, alpha_rich |
| Architecture | 4-layer MLP, ReLU, width 64/128 |
| Normalization | MinMax scaling |
| Loss | MSE with early stopping (patience=60) |

### RL environment (`src/env.py`)

- **Observation** (17-dim, normalized to [−1, 1]): gas state, OU drift signals, actual control values, actuator lag states, process outputs + derivatives/integrals, flood headroom, Pareto weight
- **Action** (4-dim): delta commands on all four control variables
- **Hard constraint**: `L_liq` is projected to guarantee `flood_fraction < 0.79`
- **Actuator lag**: first-order lags (τ = 2–5 s per variable)
- **Reward**: Pareto-weighted sum of capture rate and energy efficiency, with penalties for control roughness, capture deficit, and flooding

### PPO-LSTM policy (`train_rl.py`)

| Hyperparameter | Value |
|---|---|
| Algorithm | RecurrentPPO (MlpLstmPolicy) |
| LSTM hidden size | 128 |
| n_steps | 1024 |
| batch_size | 128 |
| Learning rate | 1e-3 (ReduceLROnPlateau) |
| gamma | 0.99 |
| clip_range | 0.2 |

---

## Results

Evaluation outputs are written to `results/`:
- `eval_results.csv` — per-episode metrics
- `comparison_*.png` — RL vs PID transient response dashboards
- `sensitivity_*.png` — OAT sensitivity heatmaps
- `sensitivity_data.csv` — raw sensitivity sweep data

TensorBoard logs are in `logs/`:

```bash
tensorboard --logdir logs/
```
