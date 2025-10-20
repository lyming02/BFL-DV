## HBDFL: Federated Learning for Electricity Theft Detection with Blockchain

An implementation of a clustered/federated learning pipeline for electricity-theft detection using a CNN+LSTM model. The system supports malicious-client simulation and two defense mechanisms, and records model hashes/metrics on a simple blockchain for auditability.

### Key Features
- CNN+LSTM classifier for weekly half-hourly energy usage sequences (length 336).
- Two FL modes:
  - FDL (classic FedAvg across all clients)
  - Cluster (clients divided into clusters with intra-cluster aggregation and defenses)
- Attack/Defense:
  - Malicious clients (label flipping and model tampering options)
  - Intra-cluster reputation-based selection (def1)
  - Inter-cluster performance drop detection with rollback (def2)
- Blockchain logging of aggregated models’ hashes and evaluation metrics.

### Repository Structure
- `generate_theft_data.py`: Build a synthetic theft dataset from raw CSVs (one week = 336 half-hour slots).
- `model.py`: `CNNLSTMNet` (Conv1D + LSTM + MLP) classifier.
- `HBDFL.py`: Main training script for FDL/Cluster modes, attacks/defenses, evaluation, and results saving.
- `configurations.py`: CLI argument definitions and defaults.
- `blockchain.py`: Minimal `Block`/`Blockchain` classes to store model hashes and metrics.
- `results/`: Directory for experiment CSV outputs (Cluster mode). FDL mode also writes under `结果/all/`.
- `.idea/`: IDE metadata (not required to run).

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- pandas
- scikit-learn
- tqdm

Install dependencies (example):
```bash
pip install torch numpy pandas scikit-learn tqdm
```

### Data Preparation
1. Place raw per-user CSVs under `data/Small LCL Data/`.
   - The script auto-detects the kWh column whose name contains both "KWH" and "hh".
2. Generate the dataset:
```bash
python generate_theft_data.py
```
This creates `theft_dataset.csv` with weekly sequences (336 values), labels (`0` normal, `1` theft), and some metadata. Synthetic theft is injected by scaling and time-window reductions.

Notes:
- If the expected kWh column is not found, the script raises a clear error showing actual columns.
- Adjust `DATA_DIR` and `OUTPUT_CSV` in `generate_theft_data.py` if needed.

### Model
`CNNLSTMNet` in `model.py` processes a 1D sequence of length 336:
- Two Conv1D+ReLU+MaxPool layers (downsample to length 336/4)
- LSTM over the downsampled sequence
- Two fully-connected layers with dropout

### Running Federated Training
The main entry point is `HBDFL.py`. It loads `theft_dataset.csv`, splits data among clients (IID or Dirichlet Non-IID), trains per-client local models, aggregates via FedAvg, and evaluates.

Basic usage (FDL mode):
```bash
python HBDFL.py --FL_TYPE FDL --num_users 30 --global_epochs 50 --local_epochs 1 --train_batch_size 64 --lr 0.001 --device cpu
```

Clustered FL with defenses:
```bash
python HBDFL.py --FL_TYPE Cluster --num_users 30 --num_clusters 3 --m_num 4 \
  --atk_mode "{'atk': 2, 'def1': 1, 'def2': 1}" --Threshold -3 --thresholdMultiplier 1.2
```

Common options (see `configurations.py` for full list and defaults):
- `--FL_TYPE {FDL,Cluster}`: Federated mode.
- `--num_users`: Total number of clients.
- `--num_clusters`: Number of clusters (Cluster mode).
- `--m_num`: Committee size for reputation evaluation (Cluster+def1).
- `--iid {True,False}`: IID split; if False, Non-IID via Dirichlet with `--alpha`.
- `--global_epochs`, `--local_epochs`, `--train_batch_size`, `--test_batch_size`, `--lr`.
- `--device {cuda:0,cpu}`: Training device.
- `--atk_mode`: Dict controlling attacks/defenses:
  - `atk`: 1=model tamper, 2=colluding malicious clients, 3=malicious cluster when imbalanced
  - `def1`: 1 enables intra-cluster reputation filtering
  - `def2`: 1 enables inter-cluster accuracy-drop detection with rollback
- `--atk_node`: List of malicious client IDs (indices).
- `--Threshold`: Accuracy drop threshold for def2 (used on a 0–100 scale in code).
- `--thresholdMultiplier`: Adaptive adjustment of `Threshold` across rounds.

Outputs:
- FDL mode: appends round metrics to `结果/all/FDL_resultsYYYYMMDD_HHMM.csv`.
- Cluster mode: appends round metrics to `results/<ratio>_m=<m>_num_users=<n>_YYYYMMDD_HHMM.csv`.

### Blockchain Logging
`blockchain.py` provides lightweight blocks:
- For each cluster update, the aggregated model state dict is hashed and stored in a block on `blockchain`.
- An additional chain (`blockchain2`) stores models and their accuracy to support rollback in def2.

### Reproducibility
Set seeds via `--seed` (used in data split and PyTorch where applicable). Due to parallelism and GPU nondeterminism, small variations may remain.

### Troubleshooting
- Column detection error in `generate_theft_data.py`:
  - Ensure raw CSVs contain a half-hourly kWh column whose name includes both "KWH" and "hh".
  - Trim header whitespace or let the script do so (it already calls `df.columns.str.strip()`).
- Directory not found:
  - Create `data/Small LCL Data/` and `results/` manually if missing. FDL mode also uses `结果/all/`.
- CUDA not used:
  - Pass `--device cuda:0` and ensure the correct PyTorch + CUDA toolkit are installed.

### Citation / Acknowledgement
If you use this code, please cite or acknowledge the repository. The dataset synthesis approach is for research and testing only and does not reflect real-world theft patterns.

### License
Specify your license here (e.g., MIT).


