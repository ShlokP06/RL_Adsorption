# RL Absorber — Deep RL Control for MEA CO₂ Capture

> A PPO-LSTM agent that replaces classical PID control in a post-combustion MEA absorber-stripper, optimizing the **capture rate / reboiler duty trade-off** in real time while enforcing hard hydraulic constraints.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## The Problem

Post-combustion capture plants absorb CO₂ from flue gas into a **monoethanolamine (MEA) solvent** in a packed absorber column, then regenerate the solvent in a stripper at high temperature. The two columns are tightly coupled: solvent flowrate, lean loading, and intercooling temperature all interact to set both **capture efficiency** and **specific reboiler duty** (GJ/t CO₂) — the dominant operating cost.

Classical PID control handles each variable independently and is tuned for a single operating point. When flue gas flowrate (G_gas) or CO₂ concentration (y_CO₂_in) shift — as they do routinely in real plants — the loops fight each other, capture rate dips, and reboiler duty spikes.

**Three additional constraints make this hard:**

- **Column flooding** — if solvent flowrate exceeds the hydraulic limit, the packed bed floods and column performance collapses. Flood fraction must stay below 0.79 at all times.
- **Lean loading** (α_lean) — controls regeneration depth. Too high wastes stripper energy; too low starves the absorber of absorption capacity.
- **Actuator dynamics** — solvent pumps, heat exchangers, and intercoolers have first-order response lags (τ = 2–5 s), so the controller must act predictively rather than reactively.

---

## Approach

A **RecurrentPPO (PPO-LSTM)** agent is trained in a Gymnasium environment backed by a fast **neural surrogate model** of the absorber-stripper. The surrogate (4-layer MLP, 6 inputs → 3 outputs) is trained on ~10,000 physics simulation points generated via Latin Hypercube Sampling — giving ~1000× speedup over calling the rate-based simulation at every RL step.

The agent simultaneously manipulates four control handles:

| Manipulated variable | Range | Physical role |
|---|---|---|
| L_liq — solvent flowrate | 2–12 kg/m²/s | Sets liquid-to-gas ratio; primary lever for CO₂ absorption |
| α_lean — lean solvent loading | 0.18–0.38 mol CO₂/mol MEA | Controls absorption driving force and regeneration duty |
| T_L_in — solvent inlet temperature | 30–55 °C | Colder solvent shifts VLE equilibrium toward higher absorption |
| T_ic — intercooling temperature | 25–50 °C | Removes reaction heat at mid-column to recover absorption capacity |

The **flooding hard constraint** is enforced at every timestep by projecting L_liq to `max_safe_L()` (Billet & Schultes hydraulic correlation) before querying the surrogate — the agent cannot violate it regardless of what the policy outputs.

The **reward** is a capture-first Pareto objective:
```
r = (η_cap/100)² + bonus·max(0, η_cap − 85)/15 − λ·(E − 3.5)/3 − penalties
```
λ is randomized per episode so the agent learns the **full Pareto frontier** between capture rate and reboiler duty — operators can shift the operating point live without retuning any parameters.

---

## Results

Benchmarked across three disturbance scenarios: step change in G_gas, step in y_CO₂_in, and simultaneous step changes.

| Metric | PPO-LSTM | PID | Winner |
|---|---|---|---|
| Peak capture rate | 98.3% | 98.5% | — |
| Mean capture (normal operating range) | ~93% | ~85% | RL |
| Recovery time after disturbance | 4–6 steps | 18–25 steps | RL |
| Time ≥ 90% capture | ~90% | ~62% | RL |
| Mean specific reboiler duty | ~3.75 GJ/t CO₂ | ~3.68 GJ/t CO₂ | PID |
| Max flood fraction | 0.750 | 0.721 | — |

In steady-state and moderate disturbances the RL agent significantly outperforms PID on capture rate and recovery speed. PID retains a slight energy advantage at the nominal operating point — the RL agent accepts a small reboiler duty penalty to maintain capture rate robustly across a wider operating envelope.

The **extreme disturbance scenarios** (large simultaneous step changes in both G_gas and y_CO₂_in) represent the hard limit of both controllers; performance converges under these conditions.

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/ShlokP06/RL_Absorber.git
cd RL_Absorber
docker compose up
```

Pre-trained models (~270 MB) are downloaded automatically on first run. Open **http://localhost:3000** for the live dashboard.

**GPU inference** (requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
USE_CUDA=1 docker compose up
```

---

### Option B — Manual setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download pre-trained models**

```bash
python download_models.py
```

Downloads `models/` (~270 MB) from the GitHub Release. Skip if training from scratch.

**3. Run the backend**

```bash
cd demo/backend
python -m uvicorn main:app --reload --port 8000
```

**4. Run the frontend** (separate terminal)

```bash
cd demo/frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

---

## Project Structure

```
RL_Absorber/
├── src/
│   ├── simulation.py        # Rate-based absorber-stripper physics (Kent-Eisenberg VLE,
│   │                        #   Billet-Schultes mass transfer, DeCoursey enhancement)
│   ├── surrogate.py         # Neural surrogate: 6 process inputs → 3 outputs (MLP)
│   └── env.py               # Gymnasium environment: 17-dim obs, 4-dim action
├── demo/
│   ├── backend/             # FastAPI server (WebSocket + REST)
│   └── frontend/            # React + Vite real-time dashboard
├── generate_data.py         # Latin Hypercube Sampling over operating space
├── merge_data.py            # Dataset deduplication and merging
├── train_surrogate.py       # MLP surrogate training
├── train_rl.py              # PPO-LSTM agent training
├── compare_controllers.py   # RL vs PID benchmark across disturbance scenarios
├── sensitivity_analysis.py  # OAT sensitivity sweeps on process inputs
├── download_models.py       # Fetch pre-trained models from GitHub Release
├── Dockerfile               # Backend container (CPU or CUDA)
└── docker-compose.yml       # Full-stack orchestration
```

---

## Architecture

### Physics model (`src/simulation.py`)

The rate-based simulation models a **60-stage packed absorber** with intercooling at stage 30 and a stripper with reboiler duty calculation:

- **VLE**: Kent-Eisenberg equilibrium model for CO₂–MEA
- **Mass transfer**: Billet & Schultes correlation for flooding and height of transfer units
- **Enhancement factor**: DeCoursey approximation for reactive CO₂ absorption into amine
- **Flooding limit**: `max_safe_L()` — bisection over L_liq to find the hydraulic limit at flood fraction = 0.79
- **Stripper**: Reboiler duty [GJ/t CO₂] as a function of rich loading and stripping temperature

### Neural surrogate (`src/surrogate.py`)

Replaces the physics model at RL training and inference time:

| | |
|---|---|
| Inputs (6) | G_gas, L_liq, y_CO₂_in, T_L_in, α_lean, T_ic |
| Outputs (3) | η_cap (capture rate %), E_specific (GJ/t CO₂), α_rich (rich loading) |
| Architecture | 4-layer MLP, ReLU, width 128, Kaiming init |
| Normalization | MinMax scaling (fitted to training data) |
| Training | MSE loss, early stopping (patience 60), target R² ≥ 0.99 on all outputs |

### RL environment (`src/env.py`)

| | |
|---|---|
| Observation (17-dim) | Gas state, OU disturbance trends, actuator positions, process outputs + derivatives/integrals, flood headroom, Pareto weight λ |
| Action (4-dim) | Δ-commands on L_liq, α_lean, T_L_in, T_ic |
| Hard constraint | L_liq projected to `max_safe_L()` before every surrogate call |
| Actuator lags | First-order dynamics, τ = 2–5 steps per variable |
| Disturbances | Ornstein-Uhlenbeck process on G_gas and y_CO₂_in (θ=0.08, σ=0.015) |

### Curriculum learning

Training introduces process complexity gradually to prevent reward collapse under large disturbances:

| Phase | Steps | Regime |
|---|---|---|
| 0 | 0 → 300k | Frozen disturbances — varied initial conditions only |
| 1 | 300k → 700k | OU drift on G_gas and y_CO₂_in |
| 2 | 700k → end | Full OU dynamics + random step changes |

### PPO-LSTM hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | RecurrentPPO (sb3-contrib) |
| Policy | MlpLstmPolicy, LSTM hidden 256 |
| Network | pi=[256,128], vf=[256,128] |
| n_steps / batch_size | 256 / 512 |
| Learning rate | 3×10⁻⁴ → 0 (linear decay) |
| γ / clip range | 0.99 / 0.2 |

---

## Training from Scratch

### 1. Generate simulation data

```bash
python generate_data.py --n 10000 --seed 101 --out data/batch1.csv
python generate_data.py --n 10000 --seed 404 --out data/batch2.csv --wide
```

`--wide` expands sampling bounds beyond nominal operating range for surrogate robustness at extremes.

### 2. Merge datasets

```bash
python merge_data.py --files "data/batch*.csv" --out data/merged_ccu.csv
```

### 3. Train the surrogate

```bash
python train_surrogate.py --data data/merged_ccu.csv --width 128
```

Saves to `models/surrogate/model.pt` and `models/surrogate/scalers.pkl`.

### 4. Train the RL agent

```bash
python train_rl.py --timesteps 2000000 --n-envs 8
```

```bash
tensorboard --logdir logs/
```

### 5. Evaluate

```bash
python train_rl.py --eval-only --model models/rl/best/best_model.zip
```

### 6. Benchmark RL vs PID

```bash
python compare_controllers.py                # all 3 disturbance scenarios
python compare_controllers.py --scenario 1   # step change in G_gas
python compare_controllers.py --scenario 2   # step change in y_CO₂_in
python compare_controllers.py --scenario 3   # simultaneous step changes
```

Outputs a 16-panel dashboard PNG and summary CSV to `results/`.

### 7. Sensitivity analysis

```bash
python sensitivity_analysis.py
```

One-at-a-time (OAT) sweeps across all six process inputs. Outputs tornado plots and heatmaps to `results/`.

---

## Demo API

Backend running at `http://localhost:8000`:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Reset both RL and PID simulations |
| POST | `/attack` | Apply hard disturbance: G_gas→1.5, y_CO₂→0.20 |
| POST | `/set_disturbance` | Manual override `{G_gas, y_CO2_in}` |
| POST | `/clear_disturbance` | Remove manual override |
| POST | `/set_lambda` | Tune Pareto weight `{lambda_energy}` live |
| POST | `/toggle_controller` | Freeze/unfreeze RL agent |
| GET | `/state` | Current process snapshot |
| GET | `/history` | Last 120 snapshots |
| WS | `/stream` | Push snapshot every 500 ms |
| GET | `/docs` | Swagger UI |
