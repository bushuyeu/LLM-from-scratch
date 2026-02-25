# Experiment Log

**W&B Project:** `ece496b-lm` (entity: `pavelbushuyeu-university-of-hawaii-system`)
**Hardware:** Lambda Labs 1x GH200 (96 GB VRAM, ARM64 Grace CPU)
**Base model:** 36M params (d_model=512, 8 layers, 8 heads, context_length=256)

---

## 1. Learning Rate Sweep (TinyStories, bs=128, 5000 steps)

| Run | LR | Val Loss | Val PPL | Runtime | State |
|---|---|---|---|---|---|
| ts-lr-3e-4-rerun | 3e-4 | 1.529 | 4.61 | 5.8 min | finished |
| ts-lr-5e-4-rerun | 5e-4 | 1.455 | 4.28 | 5.8 min | finished |
| ts-lr-1e-3 | 1e-3 | 1.377 | 3.96 | 6.0 min | finished |
| **ts-lr-2e-3** | **2e-3** | **1.349** | **3.85** | **5.9 min** | **finished** |
| ts-lr-3e-3 | 3e-3 | 1.350 | 3.86 | 5.9 min | finished |
| ts-lr-5e-3 | 5e-3 | 2.290 | 9.88 | 5.9 min | finished |
| ts-lr-1e-2 | 1e-2 | 2.529 | 12.54 | 5.9 min | finished |

*Initial runs ts-lr-3e-4 and ts-lr-5e-4 failed early (OOM/crash); rerun to completion.*

**Finding:** Best LR = 2e-3. Instability threshold between 3e-3 and 5e-3.

---

## 2. Batch Size Sweep (TinyStories, fixed LR=2e-3, 5000 steps)

| Run | BS | Val Loss | Val PPL | Runtime | State |
|---|---|---|---|---|---|
| ts-bs-1 | 1 | 3.597 | 36.49 | 2.0 min | finished |
| ts-bs-8 | 8 | 2.464 | 11.75 | 2.0 min | finished |
| ts-bs-32 | 32 | 1.824 | 6.19 | 2.4 min | finished |
| ts-bs-64 | 64 | 1.457 | 4.29 | 3.7 min | finished |
| ts-bs-128 | 128 | 1.347 | 3.85 | 5.8 min | finished |
| ts-bs-256 | 256 | 1.282 | 3.60 | 11.1 min | finished |
| ts-bs-512 | 512 | 1.228 | 3.41 | 21.2 min | finished |
| ts-bs-768 | 768 | — | — | 0.3 min | crashed (OOM) |
| ts-bs-1024 | 1024 | — | — | 0.2 min | failed (OOM) |

---

## 3. Batch Size Sweep with Tuned LR (TinyStories, 5000 steps)

| Run | BS | Tuned LR | Val Loss | Val PPL | Runtime |
|---|---|---|---|---|---|
| ts-bs-1-tuned | 1 | 1.8e-4 | 2.638 | 13.98 | 1.6 min |
| ts-bs-8-tuned | 8 | 5e-4 | 1.900 | 6.68 | 1.7 min |
| ts-bs-32-tuned | 32 | 1e-3 | 1.561 | 4.76 | 2.2 min |
| ts-bs-256-tuned | 256 | 2.8e-3 | 1.271 | 3.56 | 10.8 min |
| ts-bs-512-tuned | 512 | 3e-3 | 1.226 | 3.41 | 20.8 min |

**Finding:** Larger BS needs proportionally higher LR (sqrt scaling). Small BS (1, 8) remain less efficient even with tuned LR.

---

## 4. Ablation Studies (TinyStories, bs=128, 5000 steps, LR=2e-3)

| Run | Ablation | Val Loss | Val PPL | Runtime | State |
|---|---|---|---|---|---|
| ts-lr-2e-3 | **Baseline** | **1.349** | **3.85** | 5.9 min | finished |
| ts-ablation-no-rmsnorm | No RMSNorm (lr=2e-3) | NaN | NaN | 1.5 min | failed |
| ts-ablation-no-rmsnorm-lowlr | No RMSNorm (lr=5e-4) | NaN | NaN | 1.1 min | failed |
| ts-ablation-no-rmsnorm-vlowlr | No RMSNorm (lr=1e-4) | NaN | NaN | 1.1 min | failed |
| ts-ablation-post-norm | Post-norm | 1.972 | 7.19 | 6.1 min | finished |
| ts-ablation-no-rope-v2 | No RoPE | 1.390 | 4.01 | 6.1 min | finished |
| ts-ablation-ffn-silu-v2 | FFN SiLU (no gating) | 1.359 | 3.89 | 5.9 min | finished |

*v1 ablation runs (no-rope, ffn-silu) had bugs; v2 runs are the correct ones.*

**Findings:**
- No RMSNorm → NaN at all LRs (hard requirement)
- Post-norm → +46% loss vs baseline
- No RoPE → only +3% loss (causal mask provides implicit position info)
- SiLU vs SwiGLU → only +0.7% loss (gating is marginal at this scale)

---

## 5. OpenWebText Experiments (32K vocab, bs=128, 10K steps)

| Run | Config | Val Loss | Val PPL | Runtime | State |
|---|---|---|---|---|---|
| owt-baseline | 512/8L, fp32 | 3.799 | 44.66 | 16.1 min | finished |
| owt-bf16-flash | 512/8L, bf16+flash | 3.795 | 44.46 | 10.7 min | finished |
| ts-bf16-flash-sanity | 512/8L, bf16+flash (TS, 1K steps) | 1.878 | 6.54 | 0.6 min | finished |

**Finding:** bf16+flash matches fp32 quality with ~1.5x speedup.

---

## 6. Leaderboard / Scaling Experiments (OWT, bf16+flash)

| Run | Config | Steps | BS | LR | Val Loss | Val PPL | Runtime | State |
|---|---|---|---|---|---|---|---|---|
| **owt-v2-small-40k-v2** | **512/8L (42M, weight-tied)** | **40K** | **128** | **2e-3** | **3.738** | **42.03** | **39.0 min** | **finished** |
| owt-v2-med-640-10L | 640/10L (~70M) | 20K | 128 | 2e-3 | 4.744 | 114.86 | 25.6 min | finished |
| owt-v2-small-bs256 | 512/8L (42M) | 20K | 256 | 3e-3 | 3.816 | 45.44 | 35.9 min | finished |
| owt-scaled-768-12L | 768/12L | 20K | 64 | 1e-3 | 4.071 | 58.62 | 32.8 min | finished |
| owt-scaled-768-12L-lr3e3 | 768/12L | 20K | 64 | 3e-3 | 5.148 | 172.16 | 21.5 min | crashed |
| owt-big-1024-16L | 1024/16L | 20K | 32 | 6e-4 | 4.459 | 86.43 | 28.6 min | finished |
| owt-v2-small-40k | 512/8L | 40K | 128 | 2e-3 | 4.766 | 117.40 | 4.2 min | crashed |

**Finding:** The 42M model trained for 40K steps (1.31B tokens, 31 tokens/param) beats all larger models trained for fewer steps. Chinchilla scaling: data beats parameters within a fixed compute budget.

---

**Total experiments: 39 runs** (27 finished, 6 failed, 5 crashed, 1 sanity check)
