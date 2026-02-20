# Glossary of Terms: CS336 Assignment 1 (Basics)

This glossary contains key terms and definitions from the Assignment 1 handout, arranged chronologically as they appear in the document.

## 1. Assignment Overview
*   **Byte-Pair Encoding (BPE) Tokenizer**: A subword tokenization method that iteratively merges the most frequent pair of bytes or character sequences.
*   **Transformer Language Model (LM)**: A neural network architecture designed for processing sequential data, typically using self-attention mechanisms.
*   **Cross-Entropy Loss**: A loss function used to measure the performance of a classification model whose output is a probability value between 0 and 1.
*   **AdamW Optimizer**: A variant of the Adam optimizer that decouples weight decay from the gradient update for better regularization.
*   **Serialization**: The process of converting model and optimizer state into a format that can be stored on disk and loaded later.
*   **Perplexity**: A measurement of how well a probability distribution or probability model predicts a sample.
*   **TinyStories**: A dataset of simple children's stories used for training small language models.
*   **OpenWebText**: A larger, more complex dataset scraped from the web, used for pre-training larger models.

## 2. Tokenization
*   **Unicode Standard**: A text encoding standard that maps characters (e.g., 's', '牛') to integer **code points**.
*   **UTF-8 / UTF-16 / UTF-32**: Different encodings for converting Unicode code points into sequences of bytes. UTF-8 is the most common for web content.
*   **Subword Tokenization**: A midpoint between word-level and byte-level tokenization, trading off vocabulary size for better compression.
*   **Pre-tokenization**: A coarse-grained tokenization (e.g., splitting on whitespace or regex) used to prevent merging across punctuation or document boundaries during BPE training.
*   **Special Tokens**: Specific strings (like `<|endoftext|>`) used to encode metadata or boundaries that are never split during tokenization.

## 3. Architecture
*   **Token Embeddings**: Dense vectors representing token identities.
*   **Transformer Block**: A fundamental building block consisting of self-attention and feed-forward layers.
*   **RMSNorm (Root Mean Square Layer Normalization)**: A normalization technique that rescales activations based on their root mean square.
*   **Scaled Dot-Product Attention**: An attention mechanism that scales the dot product of queries and keys by the square root of the key dimension.
*   **Multi-Head Self-Attention (MHA)**: An attention mechanism that performs attention multiple times in parallel across different subspaces.
*   **Causal Masking**: A technique to prevent the model from "looking ahead" at future tokens during training (ensuring token $i$ only attends to positions $j \le i$).
*   **Rotary Position Embeddings (RoPE)**: A method for injecting positional information into the model by applying rotations to query and key vectors.
*   **SiLU (Sigmoid Linear Unit) / Swish**: A smooth activation function defined as $x \cdot \sigma(x)$.
*   **Gated Linear Unit (GLU)**: A gating mechanism that multiplies a linear transformation by a sigmoid-activated linear transformation.
*   **SwiGLU**: An activation function combining SiLU and GLU, used in modern architectures like Llama 3.
*   **Residual Connection**: A "shortcut" that adds the input of a layer to its output to improve gradient flow.
*   **Pre-norm vs. Post-norm**: Whether LayerNorm is applied before or after the sub-layer's main operation. Pre-norm is the modern standard for stability.
*   **FLOPs Accounting**: Calculating the floating-point operations required for model forward/backward passes.
*   **Model FLOPs Utilization (MFU)**: The ratio of observed throughput to theoretical peak hardware performance.

## 4. Training
*   **Stochastic Gradient Descent (SGD)**: The basic optimization algorithm that updates parameters using the gradient of the loss on a random batch.
*   **Learning Rate Schedule (Cosine Annealing with Warmup)**: A method for varying the learning rate over time, typically starting small (warmup), reaching a peak, and then decaying following a cosine curve.
*   **Gradient Clipping**: A technique to limit the magnitude of gradients to prevent training instability (typically by scaling gradients if their $L_2$ norm exceeds a threshold).

## 5. Training Loop & Data
*   **Data Loader**: A utility that converts a raw token sequence into a stream of training batches.
*   **Checkpointing**: Saving the model weights, optimizer state, and iteration count to allow resuming training or post-hoc study.
*   **Memory Mapping (mmap)**: A technique to map files directly to virtual memory, allowing lazy loading of large datasets without filling RAM.

## 6. Inference
*   **Temperature Scaling**: A parameter $\tau$ used to adjust the sharpness of the softmax probability distribution during sampling.
*   **Nucleus (Top-p) Sampling**: A sampling technique that selects from the smallest set of tokens whose cumulative probability exceeds a threshold $p$.

## 7. Experiments
*   **Ablation Study**: A method for understanding a system by removing or modifying individual components (e.g., removing LayerNorm) and measuring the impact.
