# Building a Transformer Language Model from Scratch

**Pavel Bushuyeu** | February 2025


**Code:** `train_bpe.py`, `tokenizer.py`, `nn.py`, `optimizer.py`, `data.py`, `checkpointing.py`, `train.py` -- all in `ece496b_basics/` and the project root.

**Model configuration (TinyStories baseline):**

| Parameter | Value |
|---|---|
| Vocab size | 10,000 (TinyStories) / 32,000 (OpenWebText) |
| Context length | 256 |
| d_model | 512 |
| Layers | 8 |
| Heads | 8 |
| d_ff | 1,408 (SwiGLU) |
| Total params | ~36M (TS) / ~42M with weight tying (OWT) |

---

## Tokenization: Bytes to Tokens

Before a model can learn anything about language, it needs a way to chop text into discrete pieces. The standard approach is Byte Pair Encoding (BPE), and the first question is: what do you start with?

### Why UTF-8 bytes?

The answer is UTF-8 bytes. UTF-8 gives you a small, fixed initial vocabulary of exactly 256 byte values, and it is the most compact encoding for English text (1 byte per ASCII character vs 2 for UTF-16 or 4 for UTF-32). Starting with UTF-16 would mean an initial vocabulary of 65,536 entries; UTF-32 would need over a million. BPE training on either would be impractical.

There is a subtlety worth knowing: UTF-8 is a variable-length encoding. The letter "a" is one byte, but the Japanese character "ko" is three bytes (`0xe3 0x81 0x93`), and an emoji like the globe can be four. A naive decoder that processes one byte at a time will crash on anything beyond ASCII. And some byte sequences are outright illegal -- for instance, `bytes([0xc0, 0x80])` is an "overlong encoding" of the null character, explicitly forbidden by the UTF-8 spec.

### Training BPE

BPE training is conceptually simple: start with 256 byte tokens, then repeatedly find the most frequent adjacent pair and merge it into a new token. Do this enough times and you get a vocabulary that compresses text efficiently.

In practice, I trained a 10K-vocab tokenizer on TinyStories and a 32K-vocab tokenizer on OpenWebText. The TinyStories tokenizer took 1,029 seconds (17 minutes) with a peak memory of just 0.08 GB. The longest tokens it learned were 15 bytes each: `' accomplishment'`, `' disappointment'`, `' responsibility'` -- frequent whole words (with the leading space from GPT-2's pre-tokenization regex) from simple children's stories.

The OpenWebText tokenizer tells a different story. Its longest token is `'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'` -- mojibake from double-encoded UTF-8, a common artifact in web-scraped data. BPE aggressively merges frequent repeated patterns, so encoding garbage becomes a single token. The OWT vocabulary also reflects its diverse source: URL fragments (`http`, `www`), political terms (`' unconstitutional'`), and technical jargon (`' cryptocurrencies'`).

Here is the key insight from profiling: the BPE merge loop itself is not the bottleneck. The `cProfile` trace shows most wall-clock time spent in `time.sleep`, `selectors.select`, and `SemLock.acquire` -- the parallel pre-tokenization pipeline and multiprocessing coordination overhead dominate.

### Compression and throughput

How good are these tokenizers? Compression ratio tells you how many raw bytes each token represents on average:

- TinyStories tokenizer on TinyStories: **4.25 bytes/token**
- OpenWebText tokenizer on OpenWebText: **4.54 bytes/token**
- TinyStories tokenizer on OpenWebText (cross-domain): **3.22 bytes/token**

The OWT tokenizer compresses slightly better because its larger vocabulary (32K vs 10K) produces longer tokens. But using the wrong tokenizer is costly -- the TinyStories tokenizer on web text drops to 3.22 bytes/token because it lacks vocabulary entries for URLs, technical terms, and diverse punctuation. Tokenizers are domain-specific.

Single-threaded encoding throughput clocks in at about 0.67 MB/s. At that rate, tokenizing The Pile (825 GB) would take roughly 342 hours. With 30 cores, you could bring it down to about 11 hours -- still not fast.

One practical note: I store token IDs as `uint16` (2 bytes) instead of `int64` (8 bytes). Both vocabularies are well under 65,536, so this is a 4x storage savings. For 541M tokens, that is ~1.1 GB vs ~4.3 GB.

---

## Architecture: What Goes Into a Transformer

I implemented every layer from scratch in `nn.py`: `Linear`, `Embedding`, `RMSNorm`, `SwiGLU`, `RoPE`, `softmax`, `scaled_dot_product_attention`, `MultiHeadSelfAttention`, `TransformerBlock`, and `TransformerLM`.

Rather than describe each one, I want to focus on what I learned from doing the resource accounting -- because that is where the real understanding comes from.

### Counting parameters

Take GPT-2 XL as a worked example: vocab=50,257, context=1,024, 48 layers, d_model=1,600, 25 heads, d_ff=6,400.

Per transformer block:

| Component | Formula | Parameters |
|---|---|---|
| RMSNorm (x2) | 2 x d_model | 3,200 |
| Q/K/V projections | 3 x d_model^2 | 7,680,000 |
| Output projection | d_model^2 | 2,560,000 |
| SwiGLU (W1, W2, W3) | 3 x d_model x d_ff | 30,720,000 |
| **Per block total** | | **40,963,200** |

Full model:

| Component | Parameters |
|---|---|
| Token embedding (V x D) | 80,411,200 |
| 48 transformer blocks | 1,966,233,600 |
| Final RMSNorm | 1,600 |
| LM head (D x V) | 80,411,200 |
| **Total** | **2,127,057,600 (~2.13B)** |

In float32, that is **8.51 GB** just for the weights.

### Where the FLOPs go

Here is where things get interesting. I computed the FLOPs for a single forward pass (batch=1, seq_len=1024) across all four GPT-2 sizes:

| Component | Small | Medium | Large | XL |
|---|---|---|---|---|
| Attn projections (QKV+O) | 16.6% | 20.0% | 21.4% | 22.3% |
| QK^T scores | 5.5% | 5.0% | 4.3% | 3.6% |
| Attention x V | 5.5% | 5.0% | 4.3% | 3.6% |
| FFN (SwiGLU) | 49.8% | 59.9% | 64.2% | 66.9% |
| LM head | 22.6% | 10.2% | 5.8% | 3.6% |
| **Total GFLOPs** | **350** | **1,033** | **2,258** | **4,513** |

The feed-forward network dominates and gets more dominant as models grow: 50% at GPT-2 Small, 67% at XL. This is because FFN cost scales as O(L x d_model x d_ff), which is roughly O(d_model^2), while attention scores scale as O(L x S^2 x d), only linear in d_model. The LM head is a fixed cost (independent of depth), so it shrinks from 23% to 4%.

But watch what happens when you crank up the context length. At S=16,384 instead of 1,024 for GPT-2 XL:

| Component | S=1,024 | S=16,384 |
|---|---|---|
| Attn projections (QKV+O) | 22.3% | 10.8% |
| QK^T + Attn x V (combined) | 7.2% | 55.2% |
| FFN (SwiGLU) | 66.9% | 32.3% |
| LM head | 3.6% | 1.8% |
| **Total** | **4.5 TFLOPs** | **149.5 TFLOPs** |

The total cost jumps 33x, almost entirely because of the quadratic attention blowup. QK^T and Attn x V each scale as O(S^2), so 16x longer context means 256x more expensive -- they go from 7% of total FLOPs to 55%. This is why efficient attention mechanisms (FlashAttention, sparse attention, linear attention) are not optional for long-context models.

### Memory accounting

Training requires far more memory than just the weights. For AdamW, you need:

- **Parameters:** 4P bytes (float32)
- **Gradients:** 4P bytes
- **Optimizer state (m and v):** 8P bytes
- **Activations:** 4 x BS x [L x (16D + HS) + 2D + 2V] bytes

The fixed cost (16P) for GPT-2 XL is ~26.2 GB. Each training sample adds ~10.5 GB of activation memory. On an 80 GB A100: max batch size is 5.

AdamW itself costs about 15 FLOPs per parameter per step (moment updates, bias correction, weight decay). For GPT-2 XL that is ~24.5 GFLOPs per optimizer step -- negligible compared to the forward/backward pass.

And the punchline: training GPT-2 XL for 400K steps with batch size 1024 on a single A100 at 50% MFU would take **14 years**. The forward pass alone is ~3,591 TFLOPs per step; with backward (2x forward), each step costs ~10,773 TFLOPs. At 9.75 TFLOP/s effective throughput, that is 1,105 seconds per step. This is why real training runs use hundreds of GPUs.

---

## Training Infrastructure

### AdamW optimizer

I implemented AdamW as a subclass of `torch.optim.Optimizer`, following the standard algorithm: exponential moving average of gradients (first moment m), exponential moving average of squared gradients (second moment v), bias correction for both, and decoupled weight decay applied directly to the parameters.

The learning rate matters more than almost any other choice. On a toy problem (minimize mean(weights^2) with decaying SGD), the difference between lr=10, lr=100, and lr=1000 is the difference between slow convergence, instant convergence, and immediate divergence to 10^18. The sweet spot is narrow and problem-dependent.

| Step | lr=10 | lr=100 | lr=1000 |
|---:|---:|---:|---:|
| 0 | 24.17 | 24.17 | 24.17 |
| 1 | 15.47 | 24.17 | 8,725.10 |
| 2 | 11.40 | 4.15 | 1,506,962.38 |
| 3 | 8.92 | 0.10 | 167,633,472 |
| 4 | 7.23 | 0.00 | 13.6B |
| 9 | 3.25 | 0.00 | 2.24 x 10^18 |

### Training loop

The training script (`train.py`) ties everything together: CLI argument parsing, memory-efficient data loading via `np.memmap`, checkpoint saving/loading, W&B logging, cosine LR schedule with warmup, gradient clipping, and sample text generation during training. Data loading (`data.py`) samples random windows from memory-mapped token arrays. Ablation flags (`--no_rope`, `--no_rmsnorm`, `--post_norm`, `--ffn_silu`, `--weight_tying`) let me test architectural variants without changing code.

### Text generation

The `generate()` function in `nn.py` implements autoregressive decoding with temperature scaling and top-p (nucleus) sampling. Temperature divides logits before softmax -- lower values sharpen the distribution toward greedy decoding, higher values flatten it toward uniform sampling. Top-p sorts probabilities descending, keeps the smallest set whose cumulative probability exceeds the threshold, zeros out the rest, renormalizes, and samples. The context window is truncated when generation exceeds the model's context length.

---

## Experiments

All experiments ran on a Lambda Labs 1x GH200 (96 GB VRAM, ARM64 Grace CPU).

### Learning rate sweep

I swept 7 learning rates on the TinyStories model (36M params, bs=128, 5000 steps):

| Learning Rate | Val Loss | Perplexity |
|---|---|---|
| 3e-4 | 1.529 | 4.61 |
| 5e-4 | 1.443 | 4.23 |
| 1e-3 | 1.377 | 3.96 |
| **2e-3** | **1.349** | **3.85** |
| 3e-3 | 1.350 | 3.86 |
| 5e-3 | 2.290 | 9.88 |
| 1e-2 | 2.529 | 12.54 |

The best learning rate is 2e-3, with 1e-3 through 3e-3 forming a plateau of near-optimal performance. The edge of stability lies between 3e-3 and 5e-3 -- at 3e-3 training is perfectly stable (loss 1.350), but at 5e-3 it oscillates and settles at 2.290. The optimal LR sits at about 40% of the instability threshold, consistent with the common wisdom that you want to be close to but safely below divergence.

### Batch size

I swept batch sizes from 1 to 512, testing both a fixed LR (2e-3) and a tuned LR using sqrt scaling:

| BS | Fixed LR (2e-3) | Tuned LR | Tuned Val Loss |
|---|---|---|---|
| 1 | 3.597 | 1.8e-4 | 2.834 |
| 8 | 2.464 | 5e-4 | 1.971 |
| 32 | 1.824 | 1e-3 | 1.638 |
| 128 | 1.349 | 2e-3 | 1.400 |
| 256 | 1.282 | 2.8e-3 | 1.271 |
| 512 | 1.228 | 3e-3 | 1.226 |

Three findings stand out. First, larger batch sizes achieve lower loss per step because they see more tokens -- but plotted against total tokens processed, the curves for bs=32 through bs=512 largely overlap, meaning sample efficiency is roughly the same. Second, small batch sizes need proportionally lower learning rates: at bs=1, tuning the LR improves loss by 21%, while at bs=512 it makes less than 1% difference. Noisy small-batch gradients amplify any learning rate excess. Third, very small batch sizes (1, 8) remain less token-efficient even with tuned LRs -- the gradient noise is both a regularizer and a waste of compute.

### Generated text

The 36M-param TinyStories model (val loss 1.349) generates remarkably coherent children's stories. Here are three samples at different settings:

**Temperature 0.8, top-p 0.9** (144 tokens):
> Once upon a time, in a big, big forest, there was a lazy cat named Tom. Tom liked to sleep all day and never wanted to play with his friends. One day, Tom saw a little bird named Tim. Tim was sad because he had no friends. Tom wanted to help Tim, so he said, "I will invite you to my house, and we can be friends." Tim was happy and said, "Yes, I will come!" At the house, Tom found a big, round ball. He told Tim, "Let's play with this ball!" Tim and Tom played with the ball and had lots of fun. They became good friends and played together every day.

**Temperature 0.5, top-p 0.95** (195 tokens):
> Once upon a time, there was a little girl named Lily. She had a big, orange ball that she loved to play with. One day, Lily went to the park with her mom and took her orange ball with her. At the park, Lily met a boy named Tom. Tom saw Lily's orange ball and wanted to play with it too. Lily asked, "Can I play with your ball, please?" Tom said, "Yes, you can play with my ball, but be careful not to lose it." Lily and Tom played with the orange ball together. They had lots of fun. After a while, they got tired and sat down under a tree. Lily's mom saw them and said, "Lily, you are very kind. I'm glad you had fun playing with the orange ball." Lily and Tom became good friends. They played together every day at the park. And Lily always remembered to be careful with her ball.

**Temperature 1.0, top-p 0.9** (195 tokens):
> Once upon a time, there was a little boy named Tim. He liked to draw with his pencil. One day, he drew a big, red apple. He showed it to his mom and said, "Look, Mom! I drew a apple!" His mom smiled and said, "That's a nice apple, Tim. But, it's dirty. Do you want to help me clean it?" Tim nodded his head and said, "Yes, Mom! I want to help!" Tim and his mom went to the kitchen. They found a big bowl and Tim got some water. He put the mix in the water to wash the apple. But the water made the apple water hot and tasty. [...] They were happy they could help their mom and make the apple taste good.

Every sample has narrative structure (setup, conflict, resolution), named characters, dialogue, and a simple moral. Lower temperature (0.5) gives more formulaic but consistently coherent text. Higher temperature (1.0) adds variety but introduces occasional logical lapses ("the water made the apple water hot and tasty"). The model is constrained to the TinyStories distribution -- it cannot produce anything outside simple children's stories.

### Ablations

I ran four architectural ablations against the baseline (pre-norm + RMSNorm + SwiGLU + RoPE), all with identical hyperparameters (lr=2e-3, bs=128, 5000 steps):

| Ablation | Val Loss | Perplexity | Delta |
|---|---|---|---|
| **Baseline** | **1.349** | **3.85** | -- |
| No RMSNorm | NaN | -- | diverged |
| Post-norm | 1.972 | 7.19 | +46% |
| No RoPE | 1.390 | 4.01 | +3% |
| FFN_SiLU (d_ff=2048) | 1.359 | 3.89 | +0.7% |

**No RMSNorm** is the most dramatic result. I replaced all RMSNorm layers with identity functions and tested at three learning rates (2e-3, 5e-4, 1e-4). The model produces NaN loss immediately at all three. Without normalization, activations accumulate through the residual stream and grow exponentially across 8 layers until they overflow to infinity. RMSNorm is not a nice-to-have -- it is a hard requirement for training deep transformers.

**Post-norm** (normalizing after the sub-layer instead of before) converges but is significantly worse. The reason is about gradient flow: in pre-norm, the residual path is clean -- the input passes through without transformation. In post-norm, normalization happens after the residual addition, so the residual stream carries raw, unnormalized sub-layer outputs. Gradients must flow through these high-variance activations, creating a bottleneck that slows optimization.

**No RoPE** is the surprise. Removing all positional encoding only hurts by 3%. The causal attention mask already provides implicit position information (each token can only see prior tokens), and statistical patterns in natural language offer further positional cues. The gap would likely widen with longer sequences, but at context length 256, the model manages well without explicit positional encoding.

**FFN_SiLU vs SwiGLU** is a near-tie. I matched parameter counts carefully: SwiGLU uses three matrices with d_ff=1,408 (2.16M params/block), while SiLU uses two matrices with d_ff=2,048 (2.10M params/block). The gating mechanism in SwiGLU provides only a 0.7% advantage at this scale. Its benefit likely compounds at larger scales.

### OpenWebText training

Same architecture but with a 32K vocabulary, trained for 10K steps on OpenWebText:

| Metric | TinyStories (5K steps) | OWT (10K steps) |
|---|---|---|
| Val Loss | 1.349 | 3.799 |
| Perplexity | 3.85 | 44.66 |
| Vocab Size | 10K | 32K |
| Training Tokens | 163.8M | 327.7M |

The higher loss is expected -- web text is far more diverse and complex. The model generates grammatically correct text with appropriate register (news/editorial tone), but coherence suffers. A sample from step 10K:

> "The reality of U.S. anti-government efforts is that the ability to profit from the continued and unregulated nature of U.S. anti-national forces has been nothing less than a sort of perseverance..."

Plausible-sounding phrases strung together without logical structure. A 58M-param model with 32K vocab is still under-parameterized for the complexity of web text -- and much of that parameter budget (33M) goes to the embedding and LM head layers rather than transformer capacity.

### Leaderboard: Optimizing under a compute budget

The final challenge: minimize OWT validation loss within 1.5 H100-equivalent hours. I implemented four optimizations:

1. **bf16 mixed precision:** `torch.amp.autocast` with bfloat16, roughly doubling throughput by using tensor cores while keeping float32 master weights.
2. **Flash Attention:** Replaced my hand-rolled attention with `torch.nn.functional.scaled_dot_product_attention`, getting O(n) memory instead of O(n^2) and substantially better speed.
3. **Weight tying:** Shared the embedding matrix with the LM head, saving ~16.4M parameters (32K x 512) and acting as a regularizer.
4. **Gradient accumulation:** Simulated larger effective batch sizes without proportionally more memory.

#### Round 1: Exploration

| Experiment | Config | Val Loss | tok/s |
|---|---|---|---|
| R1-2 | Small (512/8L) + bf16 + flash | **3.795** | **513K** |
| R1-3 | Medium (768/12L) + tying + bf16 | 4.071 | 333K |
| R1-4 | Large (1024/16L) + tying + bf16 | 4.459 | 179K |

The bf16 + flash attention combination gave a **49% throughput boost** (337K to 513K tok/s). But the most important finding was that larger models performed worse -- not because they are inherently worse, but because a fixed compute budget means fewer training tokens for bigger models. Chinchilla scaling in action: a smaller model trained on more tokens beats a larger model trained on fewer tokens.

#### Round 2: Optimized runs

| Experiment | Params | Val Loss | Perplexity | Time | Tokens |
|---|---|---|---|---|---|
| **V2-1: Small + tying, 40K steps** | **42M** | **3.738** | **42.03** | **0.65h** | **1.31B** |
| V2-2: Medium (640/10L), 20K steps | ~70M | 4.744 | 115.40 | 0.43h | 655M |
| V2-3: Small, bs=256, 20K steps | 42M | 3.816 | 45.44 | 0.60h | 1.31B |

**Best result: val loss 3.738** in 0.65 hours. Three comparisons tell the story:

**V2-1 vs V2-2:** The 42M-param model trained on 1.31B tokens (31 tokens/param) crushes the 70M-param model trained on 655M tokens (9 tokens/param). V2-2's loss of 4.744 is dramatically worse despite having more parameters. Within a fixed compute budget, data beats parameters.

**V2-1 vs V2-3:** Same model, same total tokens, but V2-3 uses batch size 256 (20K updates) while V2-1 uses batch size 128 (40K updates). V2-1 wins: 3.738 vs 3.816. More frequent, smaller gradient updates allow finer-grained optimization of the loss landscape.

**Throughput:** The small model with bf16 + flash + torch.compile achieves ~560K tok/s on the GH200.

---

## What I Learned

**Normalization is non-negotiable.** Removing RMSNorm does not degrade performance -- it makes training impossible. This was the single most eye-opening result. Pre-norm is strictly better than post-norm for the same reason: keep the residual stream clean.

**The learning rate cliff is real and steep.** Performance is nearly flat from 1e-3 to 3e-3, then falls off a cliff at 5e-3. The optimal LR sits at about 40% of the divergence threshold. This narrow-but-not-too-narrow sweet spot seems to be a general property of transformer training.

**Chinchilla scaling is not just a scaling law -- it is a practical engineering constraint.** Under a fixed compute budget, I consistently found that smaller models trained longer beat larger models. The 42M model at 1.31B tokens beat the 70M model at 655M tokens by a huge margin. Tokens per parameter is the metric that matters.

**Architectural details matter less than you think at small scale.** SwiGLU vs SiLU: 0.7% difference. RoPE vs no positional encoding: 3%. The big wins come from getting the basics right (normalization, learning rate, data volume) rather than from architectural refinements. These refinements likely compound at larger scales, which is why papers report them as significant -- but at 36M params and 5K steps, they are rounding errors.

**bf16 + FlashAttention is free performance.** A 49% throughput improvement with no loss in model quality. There is no reason not to use both.

The full codebase, training scripts, and experiment analysis are in this repository. Every layer, every optimization, every number in this document comes from code I wrote and experiments I ran.
