from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Activation functions & loss
# ---------------------------------------------------------------------------

def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax — subtract max to prevent overflow."""
    x_max = x.max(dim=dim, keepdim=True).values           # shift for stability
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish) activation: x · σ(x)."""
    return x * torch.sigmoid(x)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy loss, numerically stable via log-sum-exp.

    inputs:  (batch, vocab_size) — unnormalized logits
    targets: (batch,) — integer class indices
    """
    log_sum_exp = inputs.logsumexp(dim=-1)                 # log(Σ exp(x_j))
    correct_logits = inputs.gather(                        # x_{target}
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return (log_sum_exp - correct_logits).mean()           # -log(softmax) averaged


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """Clip combined gradient l2-norm of parameters in-place.

    If the global gradient norm exceeds max_l2_norm, all gradients are
    scaled down proportionally so the norm equals max_l2_norm.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(torch.sum(g * g) for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)                                  # rescale in-place


# ---------------------------------------------------------------------------
# nn.Module layers
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """Linear transformation (no bias): y = x W^T."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(nn.Module):
    """Embedding lookup table: maps integer token IDs to dense vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]                      # simple index lookup


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Normalizes by RMS of activations (no mean centering), then applies
    a learned element-wise scale. Simpler and faster than LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.weight).to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Computes: W2 · (SiLU(W1 · x) ⊙ W3 · x)
    where ⊙ is element-wise multiply. Two gated up-projections (W1, W3)
    followed by one down-projection (W2).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)                   # gate projection
        self.w2 = Linear(d_ff, d_model)                    # down projection
        self.w3 = Linear(d_model, d_ff)                    # up projection

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE) — Su et al., 2021
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    """Rotary Positional Embeddings.

    Encodes absolute position by rotating pairs of dimensions in Q and K
    by position-dependent angles. This makes the dot product between Q and K
    depend on their relative position, not absolute — giving relative PE
    properties while being simple to apply.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0

        # Precompute cos/sin tables — one row per position, one col per dim pair
        # θ_i = 1 / (theta^(2i/d_k))  for i = 0, 1, ..., d_k/2 - 1
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, freqs)             # (max_seq_len, d_k/2)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Apply rotation to x at given positions.

        x:               (..., seq_len, d_k)
        token_positions: (..., seq_len) integer positions
        """
        cos = self.cos_cache[token_positions]              # (..., seq, d_k/2)
        sin = self.sin_cache[token_positions]

        x1 = x[..., 0::2]                                 # even dims
        x2 = x[..., 1::2]                                 # odd dims

        # 2D rotation per dimension pair:
        # [cos θ, -sin θ] [x1]
        # [sin θ,  cos θ] [x2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.stack((out1, out2), dim=-1).flatten(-2)  # interleave back


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention (Vaswani et al., 2017).

    Q: (..., queries, d_k)    K: (..., keys, d_k)    V: (..., keys, d_v)
    mask: (..., queries, keys) boolean — True = keep, False = mask out

    Returns: (..., queries, d_v) — weighted sum of V by attention weights
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)     # (..., queries, keys)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # masked positions → -∞
    weights = softmax(scores, dim=-1)                      # normalize over keys
    return weights @ V                                     # weighted sum of values


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE (Vaswani et al., 2017).

    All heads are projected in a single matmul for efficiency, then
    split into num_heads views for the attention computation. A causal
    (lower-triangular) mask is applied so each token only attends to
    earlier positions.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads                    # per-head dimension
        self.q_proj = Linear(d_model, d_model)             # all heads in one matmul
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)        # concat heads → output
        self.rope = rope

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]

        # Project Q, K, V — shape: (..., seq, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (..., num_heads, seq, d_k) for per-head attention
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        # Apply RoPE to Q and K — encodes position via rotation
        if self.rope is not None and token_positions is not None:
            pos = token_positions.unsqueeze(-2)             # (..., 1, seq)
            pos = pos.expand(*batch_dims, self.num_heads, seq_len)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # Causal mask — each position can only attend to itself and earlier
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Concatenate heads: (..., num_heads, seq, d_k) → (..., seq, d_model)
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, -1)

        return self.output_proj(attn_out)                  # final linear projection


# ---------------------------------------------------------------------------
# Transformer Block 
# ---------------------------------------------------------------------------

class FFN_SiLU(nn.Module):
    """Standard SiLU feed-forward: W2 · SiLU(W1 · x). Two matrices, no gating."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Transformer block supporting pre-norm (default) and post-norm variants,
    optional RMSNorm removal, and FFN_SiLU alternative to SwiGLU.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_swiglu: bool = True,
    ):
        super().__init__()
        norm_cls = RMSNorm if use_rmsnorm else nn.Identity
        self.ln1 = norm_cls(d_model) if use_rmsnorm else nn.Identity()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
        self.ln2 = norm_cls(d_model) if use_rmsnorm else nn.Identity()
        self.ffn = SwiGLU(d_model, d_ff) if use_swiglu else FFN_SiLU(d_model, d_ff)
        self.post_norm = post_norm

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.post_norm:
            # Post-norm: sublayer → norm → residual
            x = x + self.ln1(self.attn(x, token_positions=token_positions))
            x = x + self.ln2(self.ffn(x))
        else:
            # Pre-norm (default): norm → sublayer → residual
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Language Model — full decoder-only LM
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """Decoder-only Transformer language model (GPT architecture).

    Pipeline: token_embeddings → N × TransformerBlock → RMSNorm → lm_head
    Outputs unnormalized logits over the vocabulary at each position.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Shared RoPE across all layers — precomputed cos/sin tables
        d_k = d_model // num_heads
        rope = RoPE(theta=rope_theta, d_k=d_k, max_seq_len=context_length)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, rope=rope,
                use_rmsnorm=use_rmsnorm, post_norm=post_norm, use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)          # project to vocab logits

    def forward(self, token_ids: Tensor) -> Tensor:
        """Forward pass: token IDs → logits.

        token_ids: (batch, seq_len) integer IDs
        returns:   (batch, seq_len, vocab_size) unnormalized logits
        """
        seq_len = token_ids.shape[-1]
        # Position IDs: 0, 1, ..., seq_len-1 — broadcast across batch
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        token_positions = token_positions.expand(token_ids.shape[0], -1)

        x = self.token_embeddings(token_ids)               # (batch, seq, d_model)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)                               # final normalization
        return self.lm_head(x)                             # (batch, seq, vocab_size)


# ---------------------------------------------------------------------------
# Text generation 
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
) -> list[int]:
    """Autoregressive text generation with temperature scaling and top-p sampling.

    model:        trained TransformerLM
    prompt_ids:   list of token IDs for the prompt
    max_tokens:   maximum number of new tokens to generate
    temperature:  softmax temperature (lower = sharper, higher = more random)
    top_p:        nucleus sampling threshold (1.0 = no filtering)
    eos_token_id: stop generation when this token is sampled (None = don't stop)

    Returns the full sequence: prompt_ids + generated tokens.
    """
    model.eval()
    device = next(model.parameters()).device
    context_length = model.context_length
    generated = list(prompt_ids)

    for _ in range(max_tokens):
        # Truncate to context window — keep the most recent tokens
        input_ids = generated[-context_length:]
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        logits = model(x)                                     # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]                        # last position logits

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Convert to probabilities
        probs = softmax(next_logits, dim=-1)

        # Top-p sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find smallest set V(p) such that sum >= top_p
            # Keep tokens where cumulative prob hasn't yet exceeded top_p,
            # plus the first token that pushes it over
            cutoff_mask = cumulative_probs - sorted_probs >= top_p
            sorted_probs[cutoff_mask] = 0.0

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum()

            # Sample from filtered distribution
            idx = torch.multinomial(sorted_probs, num_samples=1).item()
            next_token = sorted_indices[idx].item()
        else:
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated
