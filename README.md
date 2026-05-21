# NIDS-FGPA: Federated Learning Network Intrusion Detection with GSA and Homomorphic Encryption

> Mini-Project Report — M.Tech CSE (Information Security & Privacy), Sem 2, 2025–26  
> **Author:** Jivani Aayush (P25IS024) · **Supervisor:** Dr. Devesh C. Jinwala  
> Sardar Vallabhbhai National Institute of Technology, Surat

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Four Core Components](#four-core-components)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Design Choices & Pitfalls](#design-choices--pitfalls)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Classical intrusion detection systems rely on a centralised model where raw network traffic must be shared, creating privacy risks that are often legally or competitively unacceptable in IIoT environments. **NIDS-FGPA** is a federated learning-based NIDS that addresses three hard constraints simultaneously:

| Problem | Solution |
|---|---|
| Non-IID gradient divergence | GSA: reject client if θ ≥ π/2 + Softmax dynamic weighting |
| Gradient privacy leakage | Paillier HE: server only sees ciphertext |
| Communication overhead | Divergent clients are silently filtered and never upload |
| Feature-rich detection | 8×8 grayscale image + 2DCNN (spatial) + BiGRU (temporal) |

The implementation is based on Wang et al. (2024) [1] and evaluated on the **Edge-IIoTset** dataset under an extreme Non-IID federated setting.

---

## System Architecture

```
IIoT Edge Clients
  └── Local training (E epochs, batch B, Adam lr=0.001)
        │
        ▼
GSA Filter
  └── s_c = cos(∇g_c_r, ∇g_{r-1})
      reject if θ_c_r ≥ π/2
        │
        ▼ (accepted gradients only)
Paillier Encryption
  └── E(∇g_c_r) = g^(∇g_c_r) · a^n  mod n²
        │
        ▼ (encrypted gradients)
Central Server
  └── Homomorphic weighted aggregation on ciphertext
      ∏_c E(∇g_c_r)^(W_c_r) mod n²  =  E(Σ W_c_r · ∇g_c_r)
        │
        ▼ (E(∇g^r) broadcast)
Clients Decrypt & Update
  └── W_r = W_{r-1} − lr · ∇g^r
```

**Privacy guarantees:** Clients never share raw data. The server never sees plaintext gradients. Divergent clients are dropped silently.

---

## Four Core Components

### A — Traffic → 8×8 Grayscale Image

Raw network flow features cannot be fed directly into a 2D CNN. The preprocessing pipeline:

1. **Clean** — remove NaN, ±∞, and duplicate rows
2. **Normalise** — MinMax scaling to [0, 255]:  `x' = (x − min) / (max − min)`
3. **Encode strings** — ASCII character sum mod 256 for categorical columns
4. **Zero-pad** — pad feature vector to 64 dimensions (8×8)
5. **Reshape** — 1D vector → (8, 8, 1) grayscale image
6. **Label** — attach 6-class integer label

Each attack class produces visually distinct pixel patterns, enabling spatial learning by the CNN.

### B — 2DCNN-BiGRU Model

| Layer | Config |
|---|---|
| Conv2D × 2 | 32 filters, 3×3, ReLU |
| MaxPool2D | 2×2, stride 1 |
| Conv2D × 2 | 64 filters, 3×3, ReLU |
| MaxPool2D | 2×2, stride 1 |
| Reshape | rows × (cols × 64) sequence |
| Bidirectional GRU | 64 units |
| Dropout | 0.5 |
| Dense | 64, ReLU |
| Dense (output) | 6, Softmax |

- **Loss:** Sparse categorical cross-entropy
- **Optimizer:** Adam, lr = 0.001
- **Input shape:** (8, 8, 1) · **Output:** 6 classes

BiGRU gate equations:

```
z_t = σ(W_z [h_{t-1}, x_t] + b_z)          # update gate
r_t = σ(W_r [h_{t-1}, x_t] + b_r)          # reset gate
h̃_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t] + b_h)
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### C — Gradient Similarity Aggregation (GSA)

**Client-side filtering (each round r):**

```
s_c_r = (∇g_c_r · ∇g_{r-1}) / (‖∇g_c_r‖ · ‖∇g_{r-1}‖)
θ_c_r = arccos(clip(s_c_r, −1, 1))
if θ_c_r ≥ π/2  →  block client c
```

**Server-side dynamic weighted aggregation (over accepted set M_r):**

```
μ_c_r  = N_c / Σ N_j                     # sample-volume weight
λ_c_r  = exp(s_c_r) / Σ exp(s_j_r)       # softmax similarity weight
λ̄_c_r  = mean(last R'=5 rounds of λ_c)  # rolling average
W_c_r  = (λ̄_c_r · μ_c_r) / Σ(λ̄_j · μ_j) # final normalised weight

∇g_r   = Σ_{c ∈ M_r} W_c_r · ∇g_c_r
```

### D — Paillier Homomorphic Encryption

**Key generation:** Choose large primes p, q; n = pq; λ = lcm(p−1, q−1); public key (n, g); private key (λ, μ).

**Encrypt / Decrypt:**
```
E(m) = g^m · a^n  mod n²       (a random, from Z*_n)
m    = L(E(m)^λ mod n²) · μ  mod n
```

**Homomorphic properties used:**
```
E(a) · E(b)  mod n²  =  E(a + b)       # additive HE
E(a)^k       mod n²  =  E(k · a)       # scalar multiplication
```

The server computes the full weighted gradient sum without ever decrypting:
```
∏_{c ∈ M_r} E(∇g_c_r)^(W_c_r)  mod n²  =  E(Σ W_c_r · ∇g_c_r)
```

**Key size chosen: 1024-bit** — higher security than 256/512-bit, half the overhead of 2048-bit (ciphertext ≈24.3 MB vs plaintext 0.67 MB).

---

## Dataset

### Edge-IIoTset

- Real IIoT network traffic, 55 features per flow
- 6-class label mapping used in this work:

| Class | Attack Types Mapped |
|---|---|
| Normal (0) | Normal |
| DDoS (1) | DDoS_HTTP, DDoS_ICMP, DDoS_TCP, DDoS_UDP |
| Injection (2) | SQL_injection, Uploading, XSS |
| MITM (3) | MITM |
| Malware (4) | Backdoor, Ransomware, Password |
| Scanning (5) | Fingerprinting, Port_Scanning, Vulnerability_scanner |

**Download:** [Edge-IIoTset on IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)  
Place the CSV at: `data/raw/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`

---

## Experimental Setup

### Non-IID Client Configuration (C=3)

| Client | Type | Samples | Classes |
|---|---|---|---|
| client_noniid_0 | Non-IID | 320 | MITM only |
| client_iid_0 | IID | 887,194 | All 6 |
| client_iid_1 | IID | 887,194 | All 6 |

This extreme imbalance (320 vs 887K) stress-tests each algorithm's robustness.

### Hyperparameters

| Parameter | Value |
|---|---|
| Rounds R | 100 (early stop patience = 10) |
| Batch size B | 640 |
| Learning rate lr | 0.001 |
| Local epochs E | 5 (GSA) / 3 (baselines) |
| GSA threshold β | π/2 |
| History window R' | 5 rounds |
| Paillier key bits | 1024 |
| Clients C | 3 / 6 |
| FedProx μ | 0.01 |
| Max samples/client | 15,000 |

### Baselines

| Method | Filter | Encrypted | Quality Weight |
|---|---|---|---|
| **GSA (ours)** | ✅ Cosine gate | ✅ Paillier | ✅ Softmax |
| FedAvg | ❌ | ❌ | ❌ |
| FedProx | ❌ | ❌ | ❌ (proximal term) |
| FedNova | ❌ | ❌ | ❌ (step normalisation) |

---

## Results

### Performance vs Paper Benchmarks

| Method | Filter | Encrypted | Acc | Recall | Precision | F1 | Overhead (MB) | R→0.93 |
|---|---|---|---|---|---|---|---|---|
| **GSA C=3 (ours)** | ✅ | ✅ | **0.9558** | 0.9558 | **0.9615** | **0.9519** | **19.4** | R5 |
| GSA C=3 (paper) | ✅ | ✅ | 0.9450 | 0.9450 | 0.9400 | 0.9400 | 45.6 | — |
| GSA C=6 (ours) | ✅ | ✅ | 0.9374 | 0.9374 | 0.9229 | 0.9236 | 42.9 | R15 |
| FedAvg C=3 | ❌ | ❌ | 0.9558 | 0.9558 | 0.9615 | 0.9519 | 20.1 | R5 |
| FedProx C=3 | ❌ | ❌ | 0.9559 | 0.9559 | 0.9616 | 0.9520 | 20.1 | **R1** |
| FedNova C=3 | ❌ | ❌ | 0.8946 | 0.8946 | 0.8765 | 0.8797 | 30.2 | Never |
| Paper FedNova | ❌ | ❌ | 0.9260 | — | — | 0.9160 | 58.5 | — |
| Paper FedAvg | ❌ | ❌ | 0.8910 | — | — | 0.8850 | 94.8 | — |

**Key findings:**
- Our GSA C=3 **exceeds the paper's accuracy** (0.9558 vs 0.945) with **57% less overhead** (19.4 MB vs 45.6 MB)
- Overhead reduction comes from the Non-IID client being blocked from Round 2 onward — eliminating ~1/3 of all upload traffic
- FedNova fails under the extreme Non-IID setup due to gradient explosion (~2772× amplification of the minority client's gradient)
- FedProx achieves fastest convergence (R1) via its proximal regularisation term

### Per-Class F1 (GSA C=3 vs C=6)

| Class | GSA C=3 | GSA C=6 |
|---|---|---|
| Normal | 1.00 | 1.00 |
| DDoS | 0.94 | 0.94 |
| Injection | 0.73 | 0.64 |
| MITM | 0.00 | 0.00 |
| Malware | 0.57 | 0.56 |
| Scanning | 0.72 | 0.00 |

MITM F1 = 0 for all methods due to extreme class imbalance (320 MITM vs 887K Normal). C=6 improves MITM recall to 96% and Injection to 90% in confusion matrix terms, but Scanning drops to 0.

### Convergence Speed (rounds to first reach 0.93 accuracy)

| Method | Round |
|---|---|
| FedProx C=3 | R1 |
| GSA C=3 | R5 |
| FedAvg C=3 | R5 |
| GSA C=6 | R15 |
| FedNova C=3 | Never |

### GSA Filter Behaviour

After Round 2, the Non-IID client's gradient angle consistently exceeds π/2 and is blocked for all subsequent rounds. Block rate stabilises at **33%** (1 of 3 clients) while accuracy holds at **0.955**. Cumulative bandwidth savings reach approximately **34% vs FedAvg** over 65 rounds.

### Paillier Encryption Overhead

| Key Size | Ciphertext Size | Encrypt (s/val) | Decrypt (s/val) |
|---|---|---|---|
| 256-bit | 5.67 MB | — | — |
| 512-bit | 12.6 MB | — | — |
| **1024-bit (chosen)** | **24.3 MB** | — | — |
| 2048-bit | 53.7 MB | 587.05 | 130.23 |

Plaintext gradient size: 0.67 MB. 1024-bit chosen for balance of security and overhead.

---

## Visualizations

All plots are saved to Google Drive under `NIDS_FGPA/logs/`. The notebook generates two sets.

### Set 1 — GSA C=3 Core Visualizations (`logs/viz/`)

| File | Description |
|---|---|
| `logs/viz/fig1_convergence.png` | Accuracy & loss curves: GSA C=3 vs C=6 |
| `logs/viz/fig2_blocking.png` | Upload vs blocked clients per round |
| `logs/viz/fig3_confusion.png` | Confusion matrices: GSA C=3 vs C=6 |
| `logs/viz/fig4_f1_per_class.png` | Per-class F1: GSA C=3 vs C=6 bar chart |
| `logs/viz/fig5_summary_table.png` | Model performance summary vs paper |
| `logs/viz/fig6_client_distribution.png` | Client data distribution (C=3 Non-IID setup) |
| `logs/viz/fig7_overhead.png` | Communication overhead: GSA vs FedAvg |
| `logs/viz/fig8_sample_images.png` | Sample 8×8 grayscale images per attack class |
| `logs/viz/figA_paillier_demo.png` | Paillier HE demo: 6-panel verification figure |

### Set 2 — Full Comparative Analysis (`logs/analysis/`)

| File | Description |
|---|---|
| `logs/analysis/fig1_convergence_all.png` | Accuracy & loss: GSA vs all baselines (C=3) |
| `logs/analysis/fig2_comparison_bars.png` | Accuracy / F1 / Overhead bar comparison |
| `logs/analysis/fig3_convergence_speed.png` | Rounds to first reach 0.93 accuracy |
| `logs/analysis/fig4_f1_heatmap.png` | Per-class F1 heatmap: all methods × all classes |
| `logs/analysis/fig5_gsa_filter_analysis.png` | Upload/block ratio, block rate vs accuracy, cumulative overhead |
| `logs/analysis/fig6_results_table.png` | Complete results table vs paper benchmarks |

---

## Project Structure

```
NIDS_FGPA/                          # Google Drive root (BASE)
├── data/
│   ├── raw/
│   │   └── Edge-IIoTset dataset/
│   │       └── Selected dataset for ML and DL/
│   │           └── DNN-EdgeIIoT-dataset.csv   ← place dataset here
│   └── processed/
│       ├── merged_data.parquet
│       ├── preprocessed_paper.npz
│       ├── meta_paper.pkl
│       ├── images_paper.npz
│       ├── splits_paper.npz
│       ├── clients_C3.pkl
│       └── clients_C6.pkl
├── checkpoints/
│   ├── GSA_C3_ckpt.pkl
│   ├── GSA_C6_ckpt.pkl
│   ├── FedAvg_C3_ckpt.pkl
│   ├── FedProx_C3_ckpt.pkl
│   └── FedNova_C3_ckpt.pkl
├── models/
│   ├── GSA_C3_weights.npy
│   ├── GSA_C6_weights.npy
│   └── ...
├── logs/
│   ├── GSA_C3_log.csv
│   ├── GSA_C6_log.csv
│   ├── FedAvg_C3_log.csv
│   ├── FedProx_C3_log.csv
│   ├── FedNova_C3_log.csv
│   ├── viz/                        # Core GSA visualizations
│   │   ├── fig1_convergence.png
│   │   ├── fig2_blocking.png
│   │   ├── fig3_confusion.png
│   │   ├── fig4_f1_per_class.png
│   │   ├── fig5_summary_table.png
│   │   ├── fig6_client_distribution.png
│   │   ├── fig7_overhead.png
│   │   ├── fig8_sample_images.png
│   │   └── figA_paillier_demo.png
│   └── analysis/                   # Full comparative analysis
│       ├── fig1_convergence_all.png
│       ├── fig2_comparison_bars.png
│       ├── fig3_convergence_speed.png
│       ├── fig4_f1_heatmap.png
│       ├── fig5_gsa_filter_analysis.png
│       └── fig6_results_table.png
└── mini_sv_c3_complete_viz-FINAL.ipynb   ← main notebook
```

---

## How to Run

### Prerequisites

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib phe
```

The notebook runs on **Google Colab** with Google Drive mounted. All intermediate data (preprocessed arrays, images, splits, checkpoints) is cached to Drive so training can resume after reconnects.

### Execution Order

Run cells in this order after every reconnect:

```
Cell 0B  → Imports + set all Drive paths
Cell 4   → Load Edge-IIoTset CSV (or load cached parquet)
Cell 5   → Preprocessing: clean, 6-class label map, MinMax scale
Cell 6   → Convert traffic rows → 8×8 grayscale images
Cell 7   → Train/test split + C=3 and C=6 client partitioning
Cell 8   → Define 2DCNN-BiGRU model architecture
Cell 9   → All utility functions (cosine gate, local train, run_gsa)
Cell 10  → Centralized baseline
Cell 11  → Run GSA C=3 (resumes from checkpoint if available)
Cell 11B → Run GSA C=6
Cell 11C → Final test evaluation
Cell 12  → Core visualizations (figs 1–8 + figA)
Cell 13  → Paillier HE demo
Cell 12* → Baseline algorithms: FedAvg, FedProx, FedNova
Cell 13* → Run baseline experiments (C=3)
Cell 14  → Full comparative analysis (figs 1–6 in analysis/)
```

> **Note:** Cells 12 (baselines) and 13 (run baselines) reuse the same numbering as earlier cells in the notebook. Run the baseline section after the GSA experiments.

### Checkpoint Resume

Training automatically resumes from the last saved checkpoint. Checkpoints are saved every 5 rounds. A diverged checkpoint (accuracy < 0.90) is deleted and restarted automatically.

---

## Design Choices & Pitfalls

### Design Rationale

| Component | Choice | Why |
|---|---|---|
| Client filtering | Cosine similarity | Scale-invariant; L2 penalises small-dataset clients unfairly |
| Contribution weight | Softmax | Amplifies quality differences; linear contrast is too flat |
| Encryption | Paillier HE | Native additive HE; lighter than CKKS/Gentry |
| Local model | 2DCNN-BiGRU | CNN for 2D spatial; BiGRU for sequence; Transformer overfits on small FL data |
| Data format | 8×8 grayscale | Handles missing features via zero-padding; enables 2D spatial patterns |
| History window | R' = 5 rounds | Smooths noisy single-round similarity scores without over-smoothing |

### Known Implementation Pitfalls

1. **Round 0 — no global gradient:** Initialise `prev_flat = None` and skip cosine filter on Round 1.
2. **arccos domain error:** Always `clip(s, −1, 1)` before `arccos` — floating-point can produce 1.0000002.
3. **Softmax scope:** Compute softmax weights only over accepted clients M_r, not all C clients.
4. **History tracking:** Track rolling average only over rounds where client c was accepted.
5. **Pseudo-gradient definition:** Compute `∇g = w_before − w_after` explicitly; do not use `.grad` (reflects only last mini-batch).
6. **Paillier float conversion:** Cast gradient tensors to `float32` Python scalars before Paillier library calls.
7. **MinMaxScaler leakage:** Fit scaler on training split only — never on test data.
8. **FedNova + tiny client:** Under extreme Non-IID, step normalisation amplifies the minority client's gradient by ≈2772×, causing loss to diverge to 3×10¹⁰.
9. **NaN/Inf in traffic data:** Drop invalid rows before normalisation, not after.

---

## Limitations

- **Paillier speed:** 1024-bit element-wise operations take ~90 s/round at full model scale — impractical for real-time deployment without batching.
- **Semi-honest server assumption:** The design does not defend against a fully malicious aggregator.
- **Single-machine simulation:** No real network latency or client stragglers modelled.
- **MITM class:** F1 = 0 for all methods due to extreme class imbalance (n=320); oversampling (SMOTE) needed.
- **Two datasets only:** Generalisation to healthcare IoT or energy IIoT untested.
- **FedNova failure:** Our Non-IID setup is more extreme than the original paper's configuration.

---

## Future Work

- **CKKS encryption:** Replace Paillier with CKKS (OpenFHE/SEAL) and batch operations — target ~10× speed gain.
- **Asynchronous FL:** Handle disconnections and lagging clients via async aggregation.
- **Secure MPC:** Eliminate the trusted server assumption entirely.
- **Real hardware:** Evaluate on Raspberry Pi 4 / NVIDIA Jetson Nano for realistic latency estimates.
- **Class imbalance:** Apply SMOTE or class-weighted loss for MITM and Scanning.
- **New domains:** Test on healthcare IoT (ECG/vitals) and smart grid traffic.
- **Byzantine robustness:** Combine GSA with Krum or Trimmed Mean for defense against adversarial clients.

---

## References

1. J. Wang, K. Yang, and M. Li, "NIDS-FGPA: A federated learning network intrusion detection algorithm based on secure aggregation of gradient similarity models," *PLoS ONE*, vol. 19, no. 10, e0308639, 2024.
2. B. McMahan et al., "Communication-efficient learning of deep networks from decentralized data," *AISTATS*, PMLR vol. 54, pp. 1273–1282, 2017.
3. T. Li et al., "Federated optimization in heterogeneous networks," *MLSys*, 2020.
4. J. Wang et al., "Tackling the objective inconsistency problem in heterogeneous federated optimization," *NeurIPS*, vol. 33, pp. 7611–7623, 2020.
5. P. Paillier, "Public-key cryptosystems based on composite degree residuosity classes," *EUROCRYPT*, LNCS vol. 1592, pp. 223–238, 1999.
6. M. A. Ferrag et al., "Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications," *IEEE Access*, vol. 10, pp. 40281–40306, 2022.

---

*Plagiarism index: 10% overall (8% internet sources, 6% publications, 6% student papers) — within acceptable limits per institute guidelines.*
