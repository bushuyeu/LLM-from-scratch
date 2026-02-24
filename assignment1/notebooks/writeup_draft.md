# Assignment 1

## Section 2: BPE Tokenizer

> **Code files:**
> - `ece496b_basics/__init__.py` — re-exports `train_bpe`, `Tokenizer`
> - `ece496b_basics/train_bpe.py` — `train_bpe()` (Problem 2.3)
> - `ece496b_basics/tokenizer.py` — `Tokenizer` with `encode()`/`decode()` (Problem 2.4)


### 2.1 Problem (unicode1): Understanding Unicode

**(a)** What Unicode character does `chr(0)` return?

> `chr(0)` returns the **null character** (U+0000)

**(b)** How does this character's string representation (`__repr__()`) differ from its printed representation?

> `repr(chr(0))` shows `'\x00'` (an escape sequence), while `print(chr(0))` outputs nothing visible

**(c)** What happens when this character occurs in text?

> The null character is invisible.`"this is a test" + chr(0) + "string"` creates a single string of length 21, but `print()` may renders it `"this is a teststring"`

---

### 2.2 Problem (unicode2): Unicode Encodings

**(a)** Why prefer training on UTF-8 bytes over UTF-16 or UTF-32?

> UTF-8 is the most compact encoding for English/ASCII text (1 byte per character vs 2 or 4), and it gives us a small, fixed initial vocabulary of exactly 256 byte values. UTF-16 would require an initial vocabulary of 65,536 and UTF-32 would need over 1 million entries, making the BPE training impractically large.

**(b)** Why is `decode_utf8_bytes_to_str_wrong` incorrect? Give an example.

> The function decodes each byte independently: `bytes([b]).decode("utf-8")`. This works for ASCII (1 byte per character), but UTF-8 encodes some characters using multiple bytes. For example, `"こ"` encodes to 3 bytes (`b'\xe3\x81\x93'`) and `"🌍"` encodes to 4 bytes (`b'\xf0\x9f\x8c\x8d'`). Calling `bytes([0xe3]).decode("utf-8")` raises `UnicodeDecodeError` because `0xe3` alone is an incomplete multi-byte sequence.

**(c)** A two-byte sequence that does not decode to any character(s):

> `bytes([0xc0, 0x80])` — this is an "overlong encoding" of the null character, which is forbidden by the UTF-8 and hence would not decode.

---

### 2.5 Problem (train_bpe_tinystories): BPE Training on TinyStories (2 pts)

**(a)** Training stats:

> - Training time: **1029.17 seconds (17.15 minutes)**
> - Peak memory: **0.08 GB**
> - Longest tokens in vocabulary (15 bytes each):
>   - ID 7159: `' accomplishment'`
>   - ID 9142: `' disappointment'`
>   - ID 9378: `' responsibility'`
> These make sense — frequent whole words (with leading space per the pretokenization regex) get merged into single tokens in simple children's story text.

**(b)** Profile bottleneck:

> The parallel pretokenization pipeline and process synchronization (main process waiting for workers) dominates wall-clock time, rather than the core BPE merge loop. The `cProfile` trace shows most time spent in `time.sleep` / `selectors.select` / `SemLock.acquire` —  multiprocessing coordination overhead.

---

### 2.5 Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 pts)

**(a)** OWT training:

> - Trained OWT tokenizer with vocab_size=32,000
> - Longest token: `'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'`
> - This makes sense — encoding artifacts (mojibake from double-encoded UTF-8) are common in web-scraped text, and BPE aggressively merges frequent repeated patterns into single tokens.

**(b)** Compare TinyStories vs OpenWebText tokenizers:

> The TinyStories tokenizer learns simple vocabulary (`' accomplishment'`, `' disappointment'`) reflecting children's story text, while the OWT tokenizer reflects diverse web text including encoding artifacts, separator lines, URL fragments (`http`, `www`), and political/technical terms (`' unconstitutional'`, `' cryptocurrencies'`).

---

### 2.7 Problem (tokenizer_experiments): Experiments with tokenizers (4 pts)

**(a)** Compression ratios:

> - TinyStories tokenizer on TinyStories: **~4.25 bytes/token**
> - OpenWebText tokenizer on OpenWebText: **~4.54 bytes/token**
>
> The OWT tokenizer achieves a slightly higher compression ratio because it has a larger vocabulary (32K vs 10K) — more merge operations produce longer tokens, compressing text more aggressively.

**(b)** Cross-domain tokenization (OWT with TinyStories tokenizer):

> Using the TinyStories tokenizer (10K vocab) on OpenWebText yields a compression ratio of **~3.22 bytes/token**, significantly worse than the native OWT tokenizer's 4.54. The TinyStories tokenizer lacks vocabulary entries for common web text patterns (URLs, technical terms, diverse punctuation), so it falls back to shorter byte-level tokens more frequently. This demonstrates that tokenizers are domain-specific — a tokenizer trained on simple children's stories is ill-suited for complex web text.

**(c)** Throughput estimate:

> Single-threaded encoding throughput: **~0.67 MB/s**. At this rate, tokenizing The Pile (825 GB) would take approximately **825,000 / 0.67 ≈ 1,231,343 seconds ≈ 342 hours ≈ 14.25 days**. With multiprocessing (30 cores), this could theoretically be reduced to ~11.4 hours.

**(d)** Why uint16 for token IDs?

> `uint16` can represent integers 0–65,535, which covers both our TinyStories vocabulary (10K) and OpenWebText vocabulary (32K). Using uint16 (2 bytes per token) instead of int64 (8 bytes) reduces storage by 4x. For example, 541M tokens as uint16 takes ~1.1 GB vs ~4.3 GB as int64. Since our vocabulary sizes are well under 65,536, uint16 is the most memory-efficient integer type that fits all token IDs.

---

## Section 3: Transformer Language Model Architecture

> **Code file:** `ece496b_basics/nn.py`
> - `Linear` class (3.4.2), 
> - `Embedding` class (3.4.3),
> - `RMSNorm` class (3.5.1),
> - `SwiGLU` class (3.5.2),
> - `RoPE` class (3.5.3),
> - `softmax()` (3.5.4)
> - `scaled_dot_product_attention()` (3.5.4)
> - `MultiHeadSelfAttention` class (3.5.5)
> - `TransformerBlock` class (3.6), 
> - `TransformerLM` class (3.6)

### 3.6 Problem (transformer_accounting): Resource Accounting (5 pts)

GPT-2 XL config: vocab_size=50,257, context_length=1,024, num_layers=48, d_model=1,600, num_heads=25, d_ff=6,400.

**(a)** Trainable parameters and memory:

> Per Transformer block:
>
> | Component | Formula | Parameters |
> |---|---|---|
> | RMSNorm (×2) | 2 × d_model | 3,200 |
> | Q/K/V projections | 3 × d_model² | 7,680,000 |
> | Output projection | d_model² | 2,560,000 |
> | SwiGLU (W1, W2, W3) | 3 × d_model × d_ff | 30,720,000 |
> | **Per block total** | | **40,963,200** |
>
> Full model:
>
> | Component | Formula | Parameters |
> |---|---|---|
> | Token embedding | vocab_size × d_model | 80,411,200 |
> | 48 Transformer blocks | 48 × 40,963,200 | 1,966,233,600 |
> | Final RMSNorm | d_model | 1,600 |
> | LM head | d_model × vocab_size | 80,411,200 |
> | **Total** | | **2,127,057,600 ≈ 2.13B** |
>
> Memory (float32): 2,127,057,600 × 4 bytes = **8.51 GB**

**(b)** Matrix multiplies and FLOPs for a forward pass (seq_len = context_length = 1,024):

> TODO: List each matmul with dimensions and 2mnp FLOPs.
>
> Per block (batch_size × seq_len inputs):
> - Q projection: (B×S, d_model) × (d_model, d_model) → 2 × B × S × d_model² FLOPs
> - K projection: same
> - V projection: same
> - QK^T: (B×H, S, d_k) × (B×H, d_k, S) → 2 × B × H × S² × d_k
> - Attention × V: same
> - Output projection: 2 × B × S × d_model²
> - W1 (SwiGLU): 2 × B × S × d_model × d_ff
> - W3 (SwiGLU): same
> - W2 (SwiGLU): 2 × B × S × d_ff × d_model
>
> Plus embedding lookup and LM head.

**(c)** Which parts require the most FLOPs?

> The feed-forward network (SwiGLU) dominates FLOPs because it has three weight matrices (W1, W2, W3) with d_ff = 4 × d_model, making its FLOPs roughly 3× those of the attention projections. The QK^T and attention×V matmuls are relatively small when seq_len << d_model.

**(d)** Scaling across GPT-2 sizes (S=1024, B=1):

> Using the same formulas from (b), with d_ff = 4 × d_model for all variants:
>
> **GPT-2 Small** (12 layers, d_model=768, 12 heads, d_ff=3072):
> - Attn projections: 12 × 4 × 2 × 1024 × 768² = 58.0 GFLOPs
> - QK^T: 12 × 2 × 1024² × 768 = 19.3 GFLOPs
> - Attn × V: same = 19.3 GFLOPs
> - FFN (SwiGLU): 12 × 3 × 2 × 1024 × 768 × 3072 = 174.0 GFLOPs
> - LM head: 2 × 1024 × 768 × 50257 = 79.1 GFLOPs
> - **Total: 349.6 GFLOPs**
>
> **GPT-2 Medium** (24 layers, d_model=1024, 16 heads, d_ff=4096):
> - Attn projections: 24 × 4 × 2 × 1024 × 1024² = 206.2 GFLOPs
> - QK^T: 24 × 2 × 1024² × 1024 = 51.5 GFLOPs
> - Attn × V: same = 51.5 GFLOPs
> - FFN (SwiGLU): 24 × 3 × 2 × 1024 × 1024 × 4096 = 618.5 GFLOPs
> - LM head: 2 × 1024 × 1024 × 50257 = 105.4 GFLOPs
> - **Total: 1,033.1 GFLOPs**
>
> **GPT-2 Large** (36 layers, d_model=1280, 20 heads, d_ff=5120):
> - Attn projections: 36 × 4 × 2 × 1024 × 1280² = 483.2 GFLOPs
> - QK^T: 36 × 2 × 1024² × 1280 = 96.6 GFLOPs
> - Attn × V: same = 96.6 GFLOPs
> - FFN (SwiGLU): 36 × 3 × 2 × 1024 × 1280 × 5120 = 1,449.6 GFLOPs
> - LM head: 2 × 1024 × 1280 × 50257 = 131.7 GFLOPs
> - **Total: 2,257.8 GFLOPs**
>
> **GPT-2 XL** (48 layers, d_model=1600, 25 heads, d_ff=6400):
> - Attn projections: 48 × 4 × 2 × 1024 × 1600² = 1,006.6 GFLOPs
> - QK^T: 48 × 2 × 1024² × 1600 = 161.1 GFLOPs
> - Attn × V: same = 161.1 GFLOPs
> - FFN (SwiGLU): 48 × 3 × 2 × 1024 × 1600 × 6400 = 3,019.9 GFLOPs
> - LM head: 2 × 1024 × 1600 × 50257 = 164.7 GFLOPs
> - **Total: 4,513.3 GFLOPs**
>
> Summary (proportion of total FLOPs):
>
> | Component | Small | Medium | Large | XL |
> |---|---|---|---|---|
> | Attn projections (QKV+O) | 16.6% | 20.0% | 21.4% | 22.3% |
> | QK^T scores | 5.5% | 5.0% | 4.3% | 3.6% |
> | Attention × V | 5.5% | 5.0% | 4.3% | 3.6% |
> | FFN (SwiGLU) | 49.8% | 59.9% | 64.2% | 66.9% |
> | LM head | 22.6% | 10.2% | 5.8% | 3.6% |
> | **Total GFLOPs** | **350** | **1,033** | **2,258** | **4,513** |
>
> As model size increases, FFN (SwiGLU) takes up proportionally more FLOPs (50% → 67%) because its cost scales as O(L·d_model·d_ff) ∝ O(d_model²), while attention scores (QK^T, Attn×V) shrink proportionally (11% → 7%) since they scale as O(L·S²·d) — linear in d_model. The LM head also shrinks dramatically (23% → 4%) because it's a fixed cost independent of num_layers.

**(e)** GPT-2 XL with context_length = 16,384:

> **GPT-2 XL** (48 layers, d_model=1600, 25 heads, d_ff=6400, S=16,384):
> - Attn projections: 48 × 4 × 2 × 16384 × 1600² = 16.11 TFLOPs
> - QK^T: 48 × 2 × 16384² × 1600 = 41.23 TFLOPs
> - Attn × V: same = 41.23 TFLOPs
> - FFN (SwiGLU): 48 × 3 × 2 × 16384 × 1600 × 6400 = 48.32 TFLOPs
> - LM head: 2 × 16384 × 1600 × 50257 = 2.63 TFLOPs
> - **Total: 149.5 TFLOPs** (33× more than S=1024's 4.5 TFLOPs)
>
> | Component | S=1,024 | S=16,384 |
> |---|---|---|
> | Attn projections (QKV+O) | 22.3% | 10.8% |
> | QK^T scores | 3.6% | 27.6% |
> | Attention × V | 3.6% | 27.6% |
> | FFN (SwiGLU) | 66.9% | 32.3% |
> | LM head | 3.6% | 1.8% |
>
> QK^T and Attn×V scale quadratically with S (O(S²)), so at 16× longer context they become 256× more expensive each, jumping from 7.2% combined to 55.2% of total FLOPs. FFN drops from 67% to 32% — it only scales linearly with S. The overall cost increases 33× because of this quadratic attention blowup.

---

## Section 4: Training a Transformer LM

> **Code file:** `ece496b_basics/nn.py`
> - `cross_entropy()` (4.1)
> - `gradient_clipping()` (4.5)
>
> **Code file:** `ece496b_basics/optimizer.py`
> - `AdamW` class (4.3) — subclasses `torch.optim.Optimizer`
> - `get_lr_cosine_schedule()` (4.4)

### 4.1.1 Problem (adamw): Implement AdamW (2 pts)

> Code deliverable — implemented in `ece496b_basics/optimizer.py`.
> Subclasses `torch.optim.Optimizer`, which provides the `self.param_groups` / `self.state` infrastructure.
>
> `__init__` takes `params`, `lr` (α), `betas` (β₁, β₂), `eps` (ε), `weight_decay` (λ).
> `step` implements Algorithm 1 from the spec:
> 1. `m ← β₁·m + (1−β₁)·g` — first moment update
> 2. `v ← β₂·v + (1−β₂)·g²` — second moment update
> 3. Bias-correct moments: `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`
> 4. `θ ← θ − α · m̂/(√v̂ + ε)` — parameter update
> 5. `θ ← θ − α·λ·θ` — decoupled weight decay
>
> Test: `uv run pytest -k test_adamw` — PASSED

### 4.2 Problem (learning_rate_tuning): Tuning the learning rate (1 pt)

> Using the decaying SGD optimizer (θ_{t+1} = θ_t − α/√(t+1) · ∇L) on a toy problem (loss = mean(weights²), weights = 5·randn(10,10)), run for 10 steps with lr ∈ {1e1, 1e2, 1e3}:
>
> | Step | lr=10 | lr=100 | lr=1000 |
> |---:|---:|---:|---:|
> | 0 | 24.17 | 24.17 | 24.17 |
> | 1 | 15.47 | 24.17 | 8,725.10 |
> | 2 | 11.40 | 4.15 | 1,506,962.38 |
> | 3 | 8.92 | 0.10 | 167,633,472 |
> | 4 | 7.23 | 0.00 | 13.6B |
> | 5 | 5.99 | 0.00 | 857.0B |
> | 6 | 5.05 | 0.00 | 44.0T |
> | 7 | 4.32 | 0.00 | 1,892.8T |
> | 8 | 3.73 | 0.00 | 69,763.1T |
> | 9 | 3.25 | 0.00 | 2.24×10¹⁸ |
>
> - **lr=10**: Loss decays steadily but slowly (24.2 → 3.2 after 10 steps) — the effective step size is conservative, making stable but gradual progress toward the minimum.
> - **lr=100**: Loss converges rapidly to near zero by step 4 — this is the sweet spot where the optimizer takes big enough steps to converge quickly without overshooting.
> - **lr=1000**: Loss explodes immediately (24.2 → 8,725 after 1 step, then to 10¹⁸) — the learning rate massively overshoots the minimum, causing divergence.

### 4.3 Problem (adamwAccounting): Resource accounting with AdamW (2 pts)

**(a)** Peak memory (float32 = 4 bytes per value):
>
> Let B = batch_size, S = context_length, D = d_model, H = num_heads, L = num_layers, V = vocab_size, F = d_ff = 4D.
>
> **Parameters (P):**
> Per block: 4D² (QKV+O projections) + 2·D·F (FFN W1,W2) + 2D (two RMSNorms) = 4D² + 8D² + 2D = 12D² + 2D
> Full model: L·(12D² + 2D) + V·D (token embedding) + D (final RMSNorm) + V·D (lm_head)
> **P = L·(12D² + 2D) + 2VD + D**
> Memory: **4P** bytes
>
> **Gradients:** same shape as parameters = **4P** bytes
>
> **Optimizer state (AdamW):** two moment vectors (m, v), each same shape as parameters = 2 × 4P = **8P** bytes
>
> **Activations** (stored for backward pass, per layer, then model-level):
>
> Per Transformer block (×L):
> - RMSNorm inputs (×2): 2 × BSD
> - Q, K, V projections: 3 × BSD
> - Softmax output (attention weights): BHS²
> - Attention output (input to O projection): BSD
> - First residual output (input to sub-layer 2): BSD
> - RMSNorm₂ output (input to W1): BSD
> - W1 output (input to SiLU): BS·4D = 4BSD
> - SiLU output (input to W2): 4BSD
> Per-block total: 2BSD + 3BSD + BHS² + BSD + BSD + BSD + 4BSD + 4BSD = **16BSD + BHS²**
>
> Model-level:
> - Token embeddings output: BSD
> - Final RMSNorm input: BSD
> - Logits (lm_head output): BSV
> - Cross-entropy softmax probs: BSV
>
> Total activations (floats): L·(16BSD + BHS²) + 2BSD + 2BSV = **BS·[L·(16D + HS) + 2D + 2V]**
> Memory: **4 × BS·[L·(16D + HS) + 2D + 2V]** bytes
>
> **Total peak memory:**
> = 16P + 4BS·[L·(16D + HS) + 2D + 2V] bytes

**(b)** Max batch size for GPT-2 XL on 80 GB:
>
> GPT-2 XL: L=48, D=1600, H=25, S=1024, V=50257, F=6400.
>
> P = 48·(12·1600² + 2·1600) + 2·50257·1600 + 1600
> = 48·(30,720,000 + 3,200) + 160,822,400 + 1,600
> = 1,474,713,600 + 160,824,000 = **1,635,537,600 ≈ 1.64B**
>
> Fixed cost = 16P = 16 × 1,635,537,600 = 26,168,601,600 bytes ≈ **26.17 GB**
>
> Activation cost per sample (B=1):
> Per-block: 16·1024·1600 + 25·1024² = 26,214,400 + 26,214,400 = 52,428,800
> All blocks: 48 × 52,428,800 = 2,516,582,400
> Model-level: 2·1024·1600 + 2·1024·50257 = 3,276,800 + 102,926,336 = 106,203,136
> Total per sample: 2,622,785,536 floats × 4 bytes = 10,491,142,144 bytes ≈ **10.49 GB per sample**
>
> Memory (GB) ≈ **10.49 · batch_size + 26.17**
>
> 80 ≥ 10.49·B + 26.17 → B ≤ 5.13 → **B_max = 5**

**(c)** AdamW FLOPs per step:
>
> Per parameter, AdamW.step() performs (counting +, ×, ÷, √ each as 1 FLOP):
> - `m = β₁·m + (1−β₁)·g` → 3 ops (2 mul, 1 add)
> - `v = β₂·v + (1−β₂)·g²` → 4 ops (1 square, 2 mul, 1 add)
> - `m̂ = m/(1−β₁ᵗ)` → 1 op (div)
> - `v̂ = v/(1−β₂ᵗ)` → 1 op (div)
> - `p -= lr · m̂/(√v̂ + ε)` → 4 ops (sqrt, add, div, mul+sub)
> - `p -= lr·λ·p` → 2 ops (mul, sub)
>
> Total: **~15 FLOPs per parameter per step**
> For GPT-2 XL: 15 × 1.64B ≈ **24.5 GFLOPs** per optimizer step

**(d)** GPT-2 XL training time on A100 at 50% MFU (B=1024 as specified):
>
> Forward FLOPs per step with B=1, S=1024 (using d_ff=4D, 2-matrix FFN):
> Per block: 2·S·(4D² + 2·D·4D) + 2·S²·D·2 = 2S·(4D² + 8D²) + 4S²D = 24SD² + 4S²D
> All blocks: L·(24SD² + 4S²D) = 48·(24·1024·2,560,000 + 4·1,048,576·1600) = 3,342.0 GFLOPs
> LM head: 2·S·D·V = 2·1024·1600·50257 = 164.7 GFLOPs
> Total forward (B=1): **3,506.7 GFLOPs**
>
> With B=1024: 3,506.7 × 1024 = 3,590,861 GFLOPs ≈ **3,591 TFLOPs**
> Backward ≈ 2× forward: 7,182 TFLOPs
> Total per step: 3,591 + 7,182 = **10,773 TFLOPs**
>
> A100 at 50% MFU: 19.5 × 0.5 = 9.75 TFLOP/s
> Time per step: 10,773 / 9.75 = 1,105 seconds
> Total for 400K steps: 1,105 × 400,000 = 442,000,000 s ÷ 86,400 = **5,116 days ≈ 14.0 years**
>
> Training GPT-2 XL with B=1024 on a single A100 is completely impractical. Real training uses hundreds of GPUs with data/tensor parallelism to bring wall-clock time down to days.

---

## Section 5: Training Loop

> **Code files:**
> - `ece496b_basics/data.py` — `get_batch()` (5.1)
> - `ece496b_basics/checkpointing.py` — `save_checkpoint()`, `load_checkpoint()` (5.2)

### 5.1 Problem (data_loading): Implement data loading (2 pts)
> Code deliverable — implemented in `ece496b_basics/data.py`. Test: `test_get_batch`.

### 5.2 Problem (checkpointing): Implement model checkpointing (1 pt)
> Code deliverable — implemented in `ece496b_basics/checkpointing.py`. Tests: `test_checkpointing`.

### 5.3 Problem (training_together): Put it together (4 pts)
> Code deliverable — implemented in `train.py`. Features:
> - CLI args for all model/optimizer hyperparameters
> - Memory-efficient loading via `np.load(..., mmap_mode="r")`
> - Checkpoint saving/loading with `save_checkpoint`/`load_checkpoint`
> - Periodic validation loss evaluation and console logging
> - Optional W&B integration (`--wandb` flag)
> - Sample text generation during training
> - Cosine LR schedule with warmup
> - Gradient clipping
> - Ablation flags (`--no_rope`, `--no_rmsnorm`, `--post_norm`, `--ffn_silu`, `--weight_tying`)

---

## Section 6: Generating Text

> **Code file:** `ece496b_basics/nn.py` — `generate()` function
> - Temperature scaling + top-p (nucleus) sampling decoder

### Problem (decoding): Decoding (3 pts)
> Code deliverable — implemented in `ece496b_basics/nn.py` (`generate()` function).
> - Temperature scaling: divide logits by temperature before softmax
> - Top-p (nucleus) sampling: sort probabilities descending, keep smallest set whose cumulative probability exceeds `top_p`, zero out the rest, renormalize, then sample

---

## Section 7: Experiments

> **Code files:**
> - `train.py` — Main training script with CLI args, W&B logging, checkpointing (5.3)
> - `scripts/train_tinystories.sh` — SLURM job script for LR sweep (7 runs)
> - `notebooks/experiments_analysis.ipynb` — W&B data pull, loss curve plots, generation, ablation comparisons

### 7.1 Problem (experiment_log): Experiment logging (3 pts)
> W&B is integrated into `train.py` via the `--wandb` flag. Logged metrics:
> - `train/loss` — running average training loss (every `--log_interval` steps)
> - `train/lr` — current learning rate from cosine schedule
> - `train/tokens_per_sec` — throughput
> - `train/tokens_processed` — total tokens seen so far
> - `val/loss` — validation loss (every `--eval_interval` steps)
> - `val/perplexity` — exp(val_loss)
> - `sample` — generated text samples (every `--generate_interval` steps)
>
> Example: `uv run python train.py --dataset tinystories --device mps --wandb --wandb_project ece496b-lm`

### 7.2 TinyStories Experiments

**Problem (learning_rate): Tune the learning rate (3 pts)**

**(a)** Learning rate sweep:
> TODO: Run sweeps, report loss curves. Target: validation loss ≤ 1.45 on TinyStories.

**(b)** Edge of stability:
> TODO: Investigate divergence threshold vs optimal lr.

**Problem (batch_size_experiment): Batch size variations (1 pt)**
> TODO: Vary batch size from 1 to GPU memory limit. Report loss curves.

**Problem (generate): Generate text (1 pt)**
> TODO: Generate 256+ tokens from trained TinyStories model. Comment on fluency.

### 7.3 Ablations

**Problem (layer_norm_ablation): Remove RMSNorm (1 pt)**
> TODO: Train without RMSNorm, report stability issues.

**Problem (pre_norm_ablation): Post-norm Transformer (1 pt)**
> TODO: Implement post-norm, compare learning curves.

**Problem (no_pos_emb): NoPE (1 pt)**
> TODO: Remove RoPE, compare performance.

**Problem (swiglu_ablation): SwiGLU vs SiLU (1 pt)**
> TODO: Implement FFN_SiLU(x) = W2 · SiLU(W1 · x) with d_ff = 4 × d_model, compare with SwiGLU.

### 7.4 Problem (main_experiment): OWT Training (2 pts)
> TODO: Train on OpenWebText with same architecture. Report loss curves and generated text.

### 7.5 Problem (leaderboard): Leaderboard (6 pts)
> TODO: Optimize model within 1.5 H100-hours on OWT. Target: beat 5.0 validation loss baseline.
> Consider: weight tying (input/output embeddings), larger batch sizes, lr tuning, architecture tweaks from NanoGPT speedrun repo.
