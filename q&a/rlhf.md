# RLHF Q&A

## 1. Why do we need RLHF if we already have RLVR?

RLVR (Reinforcement Learning with Verifiable Rewards) only works when there is a ground-truth verifier — math proofs, code that must pass tests, factual lookups with a known answer. Most real-world tasks have no such verifier: writing style, helpfulness, safety, nuance in open-ended conversation, summarization quality, etc. RLHF fills this gap by using **human preferences as the reward signal**, which can capture subjective qualities that no automated verifier can check.

In practice the two are complementary:

- **RLVR** — use when correctness is objectively verifiable (math, code, structured extraction).
- **RLHF** — use when quality is subjective or multi-dimensional (tone, helpfulness, harmlessness, creativity).

A model trained only with RLVR may be accurate but terse, rude, or unsafe. RLHF aligns the model with broader human values and preferences that go beyond verifiable correctness.

---

## 2. What are the benefits of RLHF compared to SFT?

| Dimension | SFT | RLHF |
|---|---|---|
| **Signal type** | Imitates individual demonstrations | Optimises for comparative preferences |
| **Ceiling** | Bounded by demonstrator skill | Can exceed demonstrator skill (easier to judge than to produce) |
| **Distribution** | Learns from a fixed dataset of (prompt, gold response) pairs | Explores its own policy's distribution — trains on its own outputs |
| **Reward shaping** | Binary (correct demo or not) | Continuous reward from a learned reward model, enabling fine-grained optimization |
| **Mode coverage** | Can average over conflicting demonstrations | Preferences let the model learn which mode humans actually prefer |

Key benefits of RLHF over pure SFT:

1. **Surpassing the demonstrator.** Humans are better at ranking two outputs than writing a perfect one from scratch. RLHF exploits this asymmetry — the model can learn to produce outputs better than any single demonstration.
2. **On-policy learning.** SFT trains on a static dataset; RLHF generates responses from the current policy and gets feedback, so it directly addresses the model's actual failure modes.
3. **Reduces hallucination and harmful outputs.** Preference data naturally encodes "don't make things up" and "don't be harmful" as ranking signals, which is hard to capture in demonstration data alone.
4. **Better calibration of refusals.** SFT with refusal demonstrations tends to over-refuse; RLHF can learn nuanced boundaries.

---

## 3. What is the pipeline for preference data collection?

1. **Prompt sampling** — Curate or sample a diverse set of prompts covering the target distribution (user questions, instructions, edge cases, adversarial inputs).

2. **Response generation** — For each prompt, generate **k ≥ 2** candidate responses. These can come from:
   - The current policy (on-policy)
   - Multiple model checkpoints or different models (to increase diversity)
   - Human-written responses (for high-quality anchors)

3. **Annotation** — Human annotators compare response pairs (or rank k responses) along defined criteria (helpfulness, harmlessness, honesty). Common formats:
   - **Pairwise comparison**: given (prompt, response_A, response_B), annotator picks the better one (or ties).
   - **Likert rating**: each response scored on a scale (less common due to calibration issues across annotators).
   - **Ranking**: annotator ranks all k responses, which yields C(k,2) pairwise comparisons.

4. **Quality control** — Filter and calibrate:
   - Inter-annotator agreement checks (Cohen's κ or Fleiss' κ).
   - Gold-standard questions to detect low-quality annotators.
   - Majority voting or aggregation (e.g., Bradley-Terry model fitting) to resolve disagreements.

5. **Dataset construction** — The result is a dataset of tuples: **(prompt, chosen_response, rejected_response)**, which is used to train a reward model or directly for preference optimization (e.g., DPO).

---

## 4. Which online RL algorithm is commonly used in RLHF?

**Proximal Policy Optimization (PPO)** is the most commonly used online RL algorithm in RLHF (used in InstructGPT, early ChatGPT, etc.).

### Why PPO?

- **Stability**: PPO uses a clipped surrogate objective that prevents catastrophically large policy updates — critical when the policy is a multi-billion-parameter LLM.
- **Sample efficiency**: compared to vanilla policy gradient (REINFORCE), PPO reuses collected trajectories for multiple gradient steps.
- **Simplicity**: compared to TRPO, PPO achieves similar trust-region behavior without second-order optimization.

### PPO in the RLHF loop

```
┌─────────────────────────────────────────────────┐
│  1. Sample prompts from dataset                 │
│  2. Generate responses with current policy πθ   │
│  3. Score responses with reward model R(x, y)   │
│  4. Compute advantage estimates (GAE)           │
│  5. Update πθ with clipped PPO objective        │
│  6. Add KL penalty: r(x,y) - β·KL(πθ ‖ πref)  │
│  7. Repeat                                      │
└─────────────────────────────────────────────────┘
```

The KL divergence penalty (step 6) against a reference policy (usually the SFT checkpoint) prevents reward hacking — the model drifting too far from coherent language just to exploit the reward model.

### Notable alternatives

- **REINFORCE / RLOO** — simpler, no value head needed; used in some recent work (e.g., DeepSeek).
- **DPO (Direct Preference Optimization)** — offline method that skips the reward model entirely by reparameterizing the RL objective into a supervised loss on preference pairs. Simpler to implement but lacks on-policy exploration.
- **GRPO (Group Relative Policy Optimization)** — estimates baselines from group statistics of sampled responses, avoiding a separate value model. Used by DeepSeek-R1.
