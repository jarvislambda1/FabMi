# 🔬 Semiconductor Root Cause Analysis - Data Generation Kit

## Overview

This kit contains everything needed to generate training data for fine-tuning an LLM on semiconductor defect root cause analysis.

## Files Included

| File | Description |
|------|-------------|
| `00_MASTER_PROMPT_TEMPLATE.md` | Main prompt template and output format |
| `01_EDGE_RING.md` | Edge-ring defect patterns |
| `02_CENTER.md` | Center-concentrated defects |
| `03_DONUT.md` | Donut/ring patterns |
| `04_SCRATCH.md` | Linear scratch marks (⭐ best for correlation reasoning) |
| `05_LOC.md` | Localized clusters |
| `06_EDGE_LOC.md` | Edge-localized defects |
| `07_RANDOM.md` | Random distribution |
| `08_NEAR_FULL.md` | Catastrophic failures |
| `09_NONE_NORMAL.md` | Normal/no defect (contrast samples) |

## Quick Start

### Step 1: Open Claude/GPT-4

Use a high-quality model for generation:
- Claude Opus or Sonnet (recommended)
- GPT-4o

### Step 2: Generate Each Defect Type

For each defect type file:
1. Open the file
2. Copy the "Ready-to-Use Prompt" section
3. Paste into Claude/GPT
4. Save the JSON output

### Step 3: Combine All Outputs

Merge all JSON arrays into single training file:
```json
[
  // Edge-Ring samples...
  // Center samples...
  // ... all other types
]
```

## Target Sample Counts

| Defect Type | Samples |
|-------------|---------|
| Edge-Ring | 75 |
| Center | 75 |
| Donut | 75 |
| Scratch | 75 |
| Loc | 60 |
| Edge-Loc | 60 |
| Random | 60 |
| Near-Full | 40 |
| None | 10 |
| **Total** | **530** |

## Output Format (Alpaca Style)

```json
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}
```

## Key Differentiators

### 1. Correlation Reasoning (WOW Factor)
The **Scratch** defect type demonstrates this best:
- "Wafers 3,5,7,9 affected" → specific FOUP slots damaged
- "FOUP-A affected, FOUP-B not" → FOUP-specific, not robot

### 2. Quantitative Specs
Every sample includes specific thresholds:
- RF-hours limits (>2500 RF-hours)
- Uniformity specs (<2%)
- Leak rates (<1 mTorr/min)

### 3. Part Numbers
Real-world actionable outputs:
- P/N: 839-0127 (O-ring)
- P/N: 715-XXXX (edge ring)

### 4. Validation Criteria
How to confirm the fix worked:
- "Run qual wafer, verify uniformity <5%"
- "Leak check: <1 mTorr/min"

## Time Estimate

| Task | Time |
|------|------|
| Generate all 9 types (25 samples each) | 60-90 min |
| Review and clean data | 20 min |
| Combine into final JSON | 10 min |
| **Total** | **~2 hours** |

## Tips for Best Results

1. **Generate in batches** - 25 samples per request works well
2. **Review for uniqueness** - ensure tool IDs, lot numbers vary
3. **Check correlation clues** - especially for Scratch type
4. **Validate JSON format** - ensure parseable output

## Demo Narrative

> "We created a specialized training dataset based on real semiconductor fab troubleshooting patterns. Unlike generic LLMs that give vague advice, our fine-tuned model provides:
> 
> 1. **Specific root causes** with technical mechanisms
> 2. **Correlation reasoning** (slot patterns reveal FOUP damage)
> 3. **Actionable steps** with part numbers and specs
> 4. **Validation criteria** to confirm fixes"

---

Good luck with the hackathon! 🚀
