"""Zero-shot MATH evaluation with vLLM.

Implements problem (math_baseline) from CS336 Spring 2025 assignment 5
section 3 / ECE405 assignment 3 section 3.2:

  1. load MATH validation examples from a JSONL file
  2. format them with the r1_zero prompt template
  3. generate with a vLLM model
  4. score with `r1_zero_reward_fn` (Dr. GRPO grader)
  5. write per-example results + a metrics summary to disk

The `evaluate_vllm` function is the reusable core; later sections (EI, GRPO)
call it for validation passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROMPT_PATH_DEFAULT = Path(__file__).parent / "prompts" / "r1_zero.prompt"


def load_prompt_template(path: str | Path = PROMPT_PATH_DEFAULT) -> str:
    return Path(path).read_text()


def load_math_examples(path: str | Path) -> list[dict[str, Any]]:
    """Read a MATH JSONL file. Each row needs `problem` and `solution` fields."""
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_prompts(examples: Iterable[dict[str, Any]], template: str) -> list[str]:
    return [template.format(question=ex["problem"]) for ex in examples]


def categorize(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "format_correct_answer_correct"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "format_correct_answer_wrong"
    return "format_wrong"


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params,
    *,
    ground_truths: list[str] | None = None,
    output_path: str | Path | None = None,
    extras: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate, score, and (optionally) serialize results to disk.

    Returns the list of per-prompt result dicts so callers can compute
    aggregate metrics or re-use the generations.
    """
    if ground_truths is not None and len(ground_truths) != len(prompts):
        raise ValueError("len(ground_truths) must match len(prompts)")
    if extras is not None and len(extras) != len(prompts):
        raise ValueError("len(extras) must match len(prompts)")

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results: list[dict[str, Any]] = []
    for i, out in enumerate(outputs):
        response = out.outputs[0].text
        gt = ground_truths[i] if ground_truths is not None else None
        scores = reward_fn(response, gt) if gt is not None else {}
        record: dict[str, Any] = {
            "prompt": prompts[i],
            "response": response,
            "ground_truth": gt,
            "format_reward": scores.get("format_reward"),
            "answer_reward": scores.get("answer_reward"),
            "reward": scores.get("reward"),
            "category": (
                categorize(scores["format_reward"], scores["answer_reward"])
                if scores
                else None
            ),
            "finish_reason": out.outputs[0].finish_reason,
            "num_output_tokens": len(out.outputs[0].token_ids),
        }
        if extras is not None:
            record["meta"] = extras[i]
        results.append(record)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    cat_counts = Counter(r["category"] for r in results)
    n_correct = cat_counts["format_correct_answer_correct"]
    return {
        "n_examples": n,
        "accuracy": n_correct / n if n else 0.0,
        "format_rate": (n - cat_counts["format_wrong"]) / n if n else 0.0,
        "categories": dict(cat_counts),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/math/validation.jsonl"),
        help="MATH validation JSONL (rows with `problem`, `solution`).",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HF id or local path of the model to evaluate.",
    )
    p.add_argument(
        "--prompt-template",
        type=Path,
        default=PROMPT_PATH_DEFAULT,
        help="Prompt template with a `{question}` placeholder.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/math_baseline"),
        help="Where to write per-example JSONL and metrics summary.",
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--min-tokens", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples (debug runs).",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from vllm import LLM, SamplingParams

    args = parse_args(argv)

    examples = load_math_examples(args.data_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    template = load_prompt_template(args.prompt_template)
    prompts = format_prompts(examples, template)
    ground_truths = [ex["solution"] for ex in examples]
    extras = [
        {"level": ex.get("level"), "type": ex.get("type")} for ex in examples
    ]

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    model = LLM(
        model=args.model,
        dtype=args.dtype,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "generations.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    results = evaluate_vllm(
        model,
        r1_zero_reward_fn,
        prompts,
        sampling,
        ground_truths=ground_truths,
        output_path=results_path,
        extras=extras,
    )

    metrics = summarize(results)
    metrics["model"] = args.model
    metrics["data_path"] = str(args.data_path)
    metrics["sampling"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "stop": ["</answer>"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
