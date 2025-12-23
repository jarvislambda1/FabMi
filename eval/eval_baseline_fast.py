#!/usr/bin/env python
"""Fast baseline evaluation with reduced max_tokens."""

import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Use relative paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data" / "splits"
OUTPUT_DIR = SCRIPT_DIR / "results"

MAX_TOKENS = 800  # Reduced for faster baseline
TEMPERATURE = 0.1
SEVERITY_LEVELS = ["critical", "major", "minor", "none"]


@dataclass
class EvalResult:
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


def create_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="not-needed", timeout=120.0)


def call_api(client: OpenAI, instruction: str, input_text: str, model: str) -> Tuple[str, float]:
    prompt = f"{instruction}\n\n{input_text}"
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a semiconductor process engineer. Provide root cause analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        latency = (time.time() - start_time) * 1000
        return response.choices[0].message.content, latency
    except Exception as e:
        raise e


def extract_severity(text: str) -> str:
    text_lower = text.lower()
    for level in ['critical', 'major', 'minor']:
        if level in text_lower:
            return level
    return 'unknown'


def extract_yield_impact(text: str) -> Optional[float]:
    patterns = [r'(\d+(?:\.\d+)?)[%\s]*(?:yield|loss|impact)']
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    return None


def compute_structure_score(text: str) -> float:
    sections = [r'root cause', r'cause', r'action', r'prevention', r'severity', r'yield']
    text_lower = text.lower()
    return sum(1 for s in sections if re.search(s, text_lower)) / len(sections)


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


def evaluate_single(args) -> EvalResult:
    sample_id, sample, server_url, model = args
    client = create_client(server_url)

    try:
        predicted, latency = call_api(client, sample['instruction'], sample['input'], model)
        rouge_scores = compute_rouge(predicted, sample['output'])

        severity_gt = extract_severity(sample['output'])
        severity_pred = extract_severity(predicted)

        yield_gt = extract_yield_impact(sample['output'])
        yield_pred = extract_yield_impact(predicted)
        yield_match = abs(yield_gt - yield_pred) <= 5.0 if yield_gt and yield_pred else False

        return EvalResult(
            sample_id=sample_id,
            defect_type=sample.get('defect_type', 'unknown'),
            ground_truth=sample['output'],
            predicted=predicted,
            rouge1=rouge_scores['rouge1'],
            rouge2=rouge_scores['rouge2'],
            rougeL=rouge_scores['rougeL'],
            severity_gt=severity_gt,
            severity_pred=severity_pred,
            severity_match=(severity_gt == severity_pred),
            yield_gt=yield_gt,
            yield_pred=yield_pred,
            yield_match=yield_match,
            structure_score=compute_structure_score(predicted),
            latency_ms=latency,
        )
    except Exception as e:
        return EvalResult(
            sample_id=sample_id,
            defect_type=sample.get('defect_type', 'unknown'),
            ground_truth=sample['output'],
            predicted="",
            error=str(e),
        )


def compute_metrics(results: List[EvalResult]) -> dict:
    valid = [r for r in results if r.error is None]
    if not valid:
        return {"error": "No valid results"}

    n = len(valid)
    summary = {
        "total": len(results),
        "valid": n,
        "rouge1": round(sum(r.rouge1 for r in valid) / n, 4),
        "rouge2": round(sum(r.rouge2 for r in valid) / n, 4),
        "rougeL": round(sum(r.rougeL for r in valid) / n, 4),
        "severity_accuracy": round(sum(1 for r in valid if r.severity_match) / n, 4),
        "structure_score": round(sum(r.structure_score for r in valid) / n, 4),
        "avg_latency_ms": round(sum(r.latency_ms for r in valid) / n, 1),
    }

    yield_valid = [r for r in valid if r.yield_gt and r.yield_pred]
    if yield_valid:
        summary["yield_accuracy"] = round(sum(1 for r in yield_valid if r.yield_match) / len(yield_valid), 4)

    return summary


def main(server_url: str, model_name: str, limit: int, workers: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(DATA_DIR / "test.json") as f:
        test_samples = json.load(f)
    with open(DATA_DIR / "consolidated_all.json") as f:
        consolidated = json.load(f)

    defect_lookup = {s['input']: s.get('defect_type', 'unknown') for s in consolidated}
    for sample in test_samples:
        sample['defect_type'] = defect_lookup.get(sample['input'], 'unknown')

    test_samples = test_samples[:limit]
    print(f"Evaluating {len(test_samples)} samples with {workers} workers (max_tokens={MAX_TOKENS})...")

    tasks = [(i, s, server_url, model_name) for i, s in enumerate(test_samples)]
    results = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_single, task): task[0] for task in tasks}
        with tqdm(total=len(tasks), desc="Baseline Eval") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                valid = [r for r in results if r.error is None]
                if valid:
                    pbar.set_postfix({"ROUGE-L": f"{sum(r.rougeL for r in valid)/len(valid):.3f}"})
                pbar.update(1)

    total_time = time.time() - start_time
    results.sort(key=lambda r: r.sample_id)

    # Save
    model_safe = model_name.replace("/", "_")
    with open(OUTPUT_DIR / f"baseline_{model_safe}.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    summary = compute_metrics(results)
    summary["model"] = model_name
    summary["total_time"] = round(total_time, 1)

    with open(OUTPUT_DIR / f"summary_baseline_{model_safe}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BASELINE: {model_name}")
    print(f"{'='*60}")
    print(f"Samples: {summary['valid']}/{summary['total']}")
    print(f"ROUGE-1: {summary['rouge1']:.4f}")
    print(f"ROUGE-2: {summary['rouge2']:.4f}")
    print(f"ROUGE-L: {summary['rougeL']:.4f}")
    print(f"Severity: {summary['severity_accuracy']:.1%}")
    print(f"Yield: {summary.get('yield_accuracy', 0):.1%}")
    print(f"Structure: {summary['structure_score']:.1%}")
    print(f"Time: {total_time:.1f}s ({total_time/len(test_samples):.2f}s/sample)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="ernie_baseline")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.server_url, args.model, args.limit, args.workers)
