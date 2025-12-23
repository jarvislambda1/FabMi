#!/usr/bin/env python
"""
Semiconductor RCA Evaluation Script
Evaluates LLM accuracy on root cause analysis generation.

Usage:
    python eval_rca.py --limit 10   # test run
    python eval_rca.py              # full run (155 test samples)
"""

import json
import re
import time
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
from openai import OpenAI

# ============== Configuration ==============
import os
API_KEY = os.environ.get("NOVITA_API_KEY", "")
BASE_URL = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/openai")
DEFAULT_MODEL = "baidu/ernie-4.5-21B-a3b-thinking"

# Use relative paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data" / "splits"
OUTPUT_DIR = SCRIPT_DIR / "results"

MAX_TOKENS = 2500  # Higher for thinking models
TEMPERATURE = 0.1
REQUEST_DELAY = 2.0
MAX_RETRIES = 5
RETRY_DELAY = 5.0

# Severity levels for classification
SEVERITY_LEVELS = ["critical", "major", "minor", "none"]


# ============== Data Classes ==============
@dataclass
class EvalResult:
    """Result for a single sample evaluation"""
    sample_id: int
    defect_type: str
    ground_truth: str
    predicted: str

    # Text similarity metrics
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0

    # Task-specific metrics
    severity_gt: str = ""
    severity_pred: str = ""
    severity_match: bool = False

    yield_gt: Optional[float] = None
    yield_pred: Optional[float] = None
    yield_match: bool = False  # within ±5%

    structure_score: float = 0.0  # 0-1 based on required sections

    # Meta
    latency_ms: float = 0.0
    error: Optional[str] = None


# ============== API Client ==============
def create_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def call_api(client: OpenAI, instruction: str, input_text: str, model: str) -> Tuple[str, float]:
    """Call the LLM API and return response with latency."""

    prompt = f"{instruction}\n\n{input_text}"

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior semiconductor process engineer with 15 years of fab experience. Provide detailed root cause analysis for wafer defects."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False
            )
            latency = (time.time() - start_time) * 1000

            # Handle "thinking" models that return reasoning in reasoning_content
            message = response.choices[0].message
            content = message.content or ""

            # Check for reasoning_content (used by thinking models like ernie-thinking)
            reasoning = getattr(message, 'reasoning_content', None) or ""

            # Combine: prefer content if available, otherwise use reasoning
            final_response = content if content.strip() else reasoning

            return final_response, latency

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "400" in str(e):
                wait_time = RETRY_DELAY * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise

    raise last_error


# ============== Metric Extraction ==============
def extract_severity(text: str) -> str:
    """Extract severity level from response."""
    text_lower = text.lower()

    # Look for explicit severity mention
    patterns = [
        r'\*\*severity\*\*[:\s]*(\w+)',
        r'severity[:\s]*(\w+)',
        r'severity[:\s]*\*\*(\w+)\*\*',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            sev = match.group(1).strip()
            for level in SEVERITY_LEVELS:
                if level in sev:
                    return level

    # Fallback: check for keywords
    if 'critical' in text_lower:
        return 'critical'
    elif 'major' in text_lower:
        return 'major'
    elif 'minor' in text_lower:
        return 'minor'

    return 'unknown'


def extract_yield_impact(text: str) -> Optional[float]:
    """Extract yield impact percentage from response."""
    patterns = [
        r'\*\*yield impact\*\*[:\s]*(\d+(?:\.\d+)?)[%\s]',
        r'yield impact[:\s]*(\d+(?:\.\d+)?)[%\s]',
        r'yield[:\s]*(\d+(?:\.\d+)?)[%\s]',
        r'(\d+(?:\.\d+)?)[%\s]*yield',
        r'(\d+(?:\.\d+)?)[%\s]*(?:loss|impact)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def compute_structure_score(text: str) -> float:
    """Score based on presence of expected sections."""
    sections = [
        r'root cause',
        r'primary cause',
        r'corrective action',
        r'immediate',
        r'prevention',
        r'severity',
        r'yield',
    ]

    text_lower = text.lower()
    found = sum(1 for s in sections if re.search(s, text_lower))
    return found / len(sections)


# ============== ROUGE Computation ==============
_rouge_scorer = None

def get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return _rouge_scorer


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
    scorer = get_rouge_scorer()
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


# ============== Evaluation ==============
def evaluate_sample(
    client: OpenAI,
    sample: dict,
    sample_id: int,
    model: str
) -> EvalResult:
    """Evaluate a single sample."""

    instruction = sample['instruction']
    input_text = sample['input']
    ground_truth = sample['output']
    defect_type = sample.get('defect_type', 'unknown')

    try:
        # Call API
        predicted, latency = call_api(client, instruction, input_text, model)

        # Compute ROUGE
        rouge_scores = compute_rouge(predicted, ground_truth)

        # Extract severity
        severity_gt = extract_severity(ground_truth)
        severity_pred = extract_severity(predicted)
        severity_match = severity_gt == severity_pred

        # Extract yield impact
        yield_gt = extract_yield_impact(ground_truth)
        yield_pred = extract_yield_impact(predicted)
        yield_match = False
        if yield_gt is not None and yield_pred is not None:
            yield_match = abs(yield_gt - yield_pred) <= 5.0

        # Structure score
        structure_score = compute_structure_score(predicted)

        return EvalResult(
            sample_id=sample_id,
            defect_type=defect_type,
            ground_truth=ground_truth,
            predicted=predicted,
            rouge1=rouge_scores['rouge1'],
            rouge2=rouge_scores['rouge2'],
            rougeL=rouge_scores['rougeL'],
            severity_gt=severity_gt,
            severity_pred=severity_pred,
            severity_match=severity_match,
            yield_gt=yield_gt,
            yield_pred=yield_pred,
            yield_match=yield_match,
            structure_score=structure_score,
            latency_ms=latency,
        )

    except Exception as e:
        return EvalResult(
            sample_id=sample_id,
            defect_type=defect_type,
            ground_truth=ground_truth,
            predicted="",
            error=str(e),
        )


def compute_aggregate_metrics(results: List[EvalResult]) -> dict:
    """Compute aggregate metrics across all results."""
    total = len(results)
    valid = [r for r in results if r.error is None]
    valid_count = len(valid)

    if valid_count == 0:
        return {"total": total, "valid": 0, "error": "No valid results"}

    # Text similarity
    avg_rouge1 = sum(r.rouge1 for r in valid) / valid_count
    avg_rouge2 = sum(r.rouge2 for r in valid) / valid_count
    avg_rougeL = sum(r.rougeL for r in valid) / valid_count

    # Task metrics
    severity_matches = sum(1 for r in valid if r.severity_match)
    yield_valid = [r for r in valid if r.yield_gt is not None and r.yield_pred is not None]
    yield_matches = sum(1 for r in yield_valid if r.yield_match)

    avg_structure = sum(r.structure_score for r in valid) / valid_count
    avg_latency = sum(r.latency_ms for r in valid) / valid_count

    summary = {
        "total": total,
        "valid": valid_count,
        "errors": total - valid_count,

        # Text similarity
        "rouge1": round(avg_rouge1, 4),
        "rouge2": round(avg_rouge2, 4),
        "rougeL": round(avg_rougeL, 4),

        # Task metrics
        "severity_accuracy": round(severity_matches / valid_count, 4),
        "yield_accuracy": round(yield_matches / len(yield_valid), 4) if yield_valid else None,
        "structure_score": round(avg_structure, 4),

        # Performance
        "avg_latency_ms": round(avg_latency, 1),
    }

    # By defect type
    defect_types = set(r.defect_type for r in valid)
    summary["by_defect_type"] = {}

    for dt in sorted(defect_types):
        dt_results = [r for r in valid if r.defect_type == dt]
        if dt_results:
            summary["by_defect_type"][dt] = {
                "count": len(dt_results),
                "rouge1": round(sum(r.rouge1 for r in dt_results) / len(dt_results), 4),
                "rougeL": round(sum(r.rougeL for r in dt_results) / len(dt_results), 4),
                "severity_acc": round(sum(1 for r in dt_results if r.severity_match) / len(dt_results), 4),
                "structure": round(sum(r.structure_score for r in dt_results) / len(dt_results), 4),
            }

    return summary


def print_summary(summary: dict, model: str):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION: {model}")
    print("=" * 60)

    print(f"\nSamples: {summary['valid']}/{summary['total']} valid ({summary['errors']} errors)")

    print(f"\n{'TEXT SIMILARITY'}")
    print("-" * 40)
    print(f"  ROUGE-1:  {summary.get('rouge1', 'N/A'):.4f}")
    print(f"  ROUGE-2:  {summary.get('rouge2', 'N/A'):.4f}")
    print(f"  ROUGE-L:  {summary.get('rougeL', 'N/A'):.4f}")

    print(f"\n{'TASK ACCURACY'}")
    print("-" * 40)
    print(f"  Severity Match:    {summary.get('severity_accuracy', 0):.1%}")
    yield_acc = summary.get('yield_accuracy')
    print(f"  Yield Impact (±5%): {yield_acc:.1%}" if yield_acc else "  Yield Impact: N/A")
    print(f"  Structure Score:   {summary.get('structure_score', 0):.1%}")

    print(f"\n{'PERFORMANCE'}")
    print("-" * 40)
    print(f"  Avg Latency: {summary.get('avg_latency_ms', 0):.0f}ms")

    print(f"\n{'BY DEFECT TYPE'}")
    print("-" * 40)
    for dt, metrics in summary.get("by_defect_type", {}).items():
        print(f"\n  {dt} (n={metrics['count']}):")
        print(f"    ROUGE-L: {metrics['rougeL']:.3f}, Severity: {metrics['severity_acc']:.1%}, Structure: {metrics['structure']:.1%}")


def main(
    limit: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    resume_from: Optional[str] = None,
):
    """Run the evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"Data: {DATA_DIR}")

    # Load test data with defect types from consolidated file
    consolidated_file = DATA_DIR / "consolidated_all.json"
    test_file = DATA_DIR / "test.json"

    # Load test samples
    with open(test_file) as f:
        test_samples = json.load(f)

    # Load consolidated to get defect_type mapping
    with open(consolidated_file) as f:
        consolidated = json.load(f)

    # Create lookup by input text
    defect_lookup = {s['input']: s.get('defect_type', 'unknown') for s in consolidated}

    # Add defect_type to test samples
    for sample in test_samples:
        sample['defect_type'] = defect_lookup.get(sample['input'], 'unknown')

    print(f"Loaded {len(test_samples)} test samples")

    if limit:
        test_samples = test_samples[:limit]
        print(f"Limited to {limit} samples")

    # Resume support
    results = []
    completed_ids = set()
    model_safe = model.replace("/", "_")
    results_file = OUTPUT_DIR / f"baseline_{model_safe}.json"

    if resume_from and Path(resume_from).exists():
        with open(resume_from) as f:
            existing = json.load(f)
            results = [EvalResult(**r) for r in existing]
            completed_ids = {r.sample_id for r in results}
            print(f"Resuming from {len(completed_ids)} completed samples")

    # Filter out completed
    remaining = [(i, s) for i, s in enumerate(test_samples) if i not in completed_ids]
    print(f"Remaining samples: {len(remaining)}")

    if not remaining:
        print("All samples already processed!")
        summary = compute_aggregate_metrics(results)
        print_summary(summary, model)
        return

    # Create client
    client = create_client()

    # Process samples
    pbar = tqdm(remaining, desc="Evaluating", unit="sample")
    for i, (sample_id, sample) in enumerate(pbar):
        pbar.set_postfix_str(f"{sample['defect_type'][:10]}")

        result = evaluate_sample(client, sample, sample_id, model)
        results.append(result)

        # Update progress
        valid = [r for r in results if r.error is None]
        if valid:
            current_rougeL = sum(r.rougeL for r in valid) / len(valid)
            pbar.set_description(f"Eval (ROUGE-L: {current_rougeL:.3f})")

        # Save intermediate results every 5 samples
        if (i + 1) % 5 == 0:
            with open(results_file, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)

        time.sleep(REQUEST_DELAY)

    # Save final results
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Compute and save summary
    summary = compute_aggregate_metrics(results)
    summary["model"] = model
    summary_file = OUTPUT_DIR / f"summary_{model_safe}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, model)
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM on semiconductor RCA")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--resume", type=str, help="Resume from existing results file")

    args = parser.parse_args()

    main(limit=args.limit, model=args.model, resume_from=args.resume)
