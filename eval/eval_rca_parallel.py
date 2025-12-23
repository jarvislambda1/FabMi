#!/usr/bin/env python
"""
Semiconductor RCA Evaluation Script (Parallel Version)
Evaluates LLM accuracy on root cause analysis generation with parallel requests.

Usage:
    python eval_rca_parallel.py --limit 9    # test run
    python eval_rca_parallel.py              # full run (155 test samples)
"""

import json
import re
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
import aiohttp

# ============== Configuration ==============
import os
API_KEY = os.environ.get("NOVITA_API_KEY", "")
BASE_URL = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/openai")
DEFAULT_MODEL = "baidu/ernie-4.5-21B-a3b-thinking"

# Use relative paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data" / "splits"
OUTPUT_DIR = SCRIPT_DIR / "results"

MAX_TOKENS = 2500
TEMPERATURE = 0.1

# Rate limiting: 10 RPM = 1 request per 6 seconds
RATE_LIMIT_INTERVAL = 6.5  # slightly conservative
MAX_WORKERS = 4  # concurrent requests in flight

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 10.0

# Severity levels
SEVERITY_LEVELS = ["critical", "major", "minor", "none"]


# ============== Data Classes ==============
@dataclass
class EvalResult:
    """Result for a single sample evaluation"""
    sample_id: int
    defect_type: str
    ground_truth: str
    predicted: str
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    severity_gt: str = ""
    severity_pred: str = ""
    severity_match: bool = False
    yield_gt: Optional[float] = None
    yield_pred: Optional[float] = None
    yield_match: bool = False
    structure_score: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None


# ============== Rate Limiter ==============
class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, interval: float):
        self.interval = interval
        self.last_request = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request = time.time()


# ============== Metric Extraction ==============
def extract_severity(text: str) -> str:
    text_lower = text.lower()
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
    if 'critical' in text_lower:
        return 'critical'
    elif 'major' in text_lower:
        return 'major'
    elif 'minor' in text_lower:
        return 'minor'
    return 'unknown'


def extract_yield_impact(text: str) -> Optional[float]:
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
    sections = [
        r'root cause', r'primary cause', r'corrective action',
        r'immediate', r'prevention', r'severity', r'yield',
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
    scorer = get_rouge_scorer()
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


# ============== Async API Call ==============
async def call_api_async(
    client: AsyncOpenAI,
    instruction: str,
    input_text: str,
    model: str,
    rate_limiter: RateLimiter
) -> Tuple[str, float]:
    """Call the LLM API asynchronously with rate limiting."""

    prompt = f"{instruction}\n\n{input_text}"

    await rate_limiter.acquire()

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            response = await client.chat.completions.create(
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
            )
            latency = (time.time() - start_time) * 1000

            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, 'reasoning_content', None) or ""
            final_response = content if content.strip() else reasoning

            return final_response, latency

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise

    raise last_error


async def evaluate_sample_async(
    client: AsyncOpenAI,
    sample: dict,
    sample_id: int,
    model: str,
    rate_limiter: RateLimiter
) -> EvalResult:
    """Evaluate a single sample asynchronously."""

    instruction = sample['instruction']
    input_text = sample['input']
    ground_truth = sample['output']
    defect_type = sample.get('defect_type', 'unknown')

    try:
        predicted, latency = await call_api_async(
            client, instruction, input_text, model, rate_limiter
        )

        rouge_scores = compute_rouge(predicted, ground_truth)
        severity_gt = extract_severity(ground_truth)
        severity_pred = extract_severity(predicted)
        severity_match = severity_gt == severity_pred
        yield_gt = extract_yield_impact(ground_truth)
        yield_pred = extract_yield_impact(predicted)
        yield_match = False
        if yield_gt is not None and yield_pred is not None:
            yield_match = abs(yield_gt - yield_pred) <= 5.0
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


async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    model: str,
    rate_limiter: RateLimiter,
    results: List[EvalResult],
    progress: dict
):
    """Worker that processes samples from queue."""
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        sample_id, sample = item
        result = await evaluate_sample_async(
            client, sample, sample_id, model, rate_limiter
        )
        results.append(result)

        progress['completed'] += 1
        valid = [r for r in results if r.error is None]
        if valid:
            avg_rouge = sum(r.rougeL for r in valid) / len(valid)
            print(f"[{progress['completed']}/{progress['total']}] "
                  f"Worker {worker_id} | {sample['defect_type'][:10]} | "
                  f"ROUGE-L: {avg_rouge:.3f}")

        queue.task_done()


def compute_aggregate_metrics(results: List[EvalResult]) -> dict:
    """Compute aggregate metrics across all results."""
    total = len(results)
    valid = [r for r in results if r.error is None]
    valid_count = len(valid)

    if valid_count == 0:
        return {"total": total, "valid": 0, "error": "No valid results"}

    avg_rouge1 = sum(r.rouge1 for r in valid) / valid_count
    avg_rouge2 = sum(r.rouge2 for r in valid) / valid_count
    avg_rougeL = sum(r.rougeL for r in valid) / valid_count

    severity_matches = sum(1 for r in valid if r.severity_match)
    yield_valid = [r for r in valid if r.yield_gt is not None and r.yield_pred is not None]
    yield_matches = sum(1 for r in yield_valid if r.yield_match)

    avg_structure = sum(r.structure_score for r in valid) / valid_count
    avg_latency = sum(r.latency_ms for r in valid) / valid_count

    summary = {
        "total": total,
        "valid": valid_count,
        "errors": total - valid_count,
        "rouge1": round(avg_rouge1, 4),
        "rouge2": round(avg_rouge2, 4),
        "rougeL": round(avg_rougeL, 4),
        "severity_accuracy": round(severity_matches / valid_count, 4),
        "yield_accuracy": round(yield_matches / len(yield_valid), 4) if yield_valid else None,
        "structure_score": round(avg_structure, 4),
        "avg_latency_ms": round(avg_latency, 1),
    }

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


def print_summary(summary: dict, model: str, elapsed: float):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION: {model}")
    print("=" * 60)

    print(f"\nSamples: {summary['valid']}/{summary['total']} valid ({summary['errors']} errors)")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

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
        print(f"  {dt} (n={metrics['count']}): ROUGE-L={metrics['rougeL']:.3f}, "
              f"Sev={metrics['severity_acc']:.0%}, Struct={metrics['structure']:.0%}")


async def main_async(
    limit: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    num_workers: int = MAX_WORKERS,
):
    """Run the parallel evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"Workers: {num_workers}")
    print(f"Rate limit: 1 request per {RATE_LIMIT_INTERVAL}s")

    # Load data
    consolidated_file = DATA_DIR / "consolidated_all.json"
    test_file = DATA_DIR / "test.json"

    with open(test_file) as f:
        test_samples = json.load(f)

    with open(consolidated_file) as f:
        consolidated = json.load(f)

    defect_lookup = {s['input']: s.get('defect_type', 'unknown') for s in consolidated}
    for sample in test_samples:
        sample['defect_type'] = defect_lookup.get(sample['input'], 'unknown')

    print(f"Loaded {len(test_samples)} test samples")

    if limit:
        test_samples = test_samples[:limit]
        print(f"Limited to {limit} samples")

    # Setup
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    rate_limiter = RateLimiter(RATE_LIMIT_INTERVAL)

    queue = asyncio.Queue()
    for i, sample in enumerate(test_samples):
        queue.put_nowait((i, sample))

    results: List[EvalResult] = []
    progress = {'completed': 0, 'total': len(test_samples)}

    print(f"\nStarting evaluation with {num_workers} workers...")
    start_time = time.time()

    # Create and run workers
    workers = [
        asyncio.create_task(
            worker(i, queue, client, model, rate_limiter, results, progress)
        )
        for i in range(num_workers)
    ]

    await asyncio.gather(*workers)

    elapsed = time.time() - start_time

    # Sort results by sample_id
    results.sort(key=lambda r: r.sample_id)

    # Save results
    model_safe = model.replace("/", "_")
    results_file = OUTPUT_DIR / f"baseline_{model_safe}_parallel.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Compute and save summary
    summary = compute_aggregate_metrics(results)
    summary["model"] = model
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["workers"] = num_workers

    summary_file = OUTPUT_DIR / f"summary_{model_safe}_parallel.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, model, elapsed)
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")


def main(limit: Optional[int] = None, model: str = DEFAULT_MODEL, workers: int = MAX_WORKERS):
    asyncio.run(main_async(limit=limit, model=model, num_workers=workers))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel evaluation of LLM on semiconductor RCA")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers")

    args = parser.parse_args()

    main(limit=args.limit, model=args.model, workers=args.workers)
