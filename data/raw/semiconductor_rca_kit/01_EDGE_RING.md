# Defect Type: EDGE-RING

## Visual Pattern
Defects forming ring pattern at wafer edge, 5-15mm from periphery

## Known Root Causes
- Aged O-ring seal in etch chamber (typical life: 2500 RF-hours)
- Plasma confinement ring erosion affecting edge plasma density
- Edge bead removal (EBR) nozzle misalignment or clog
- Clamp ring contamination or wear
- Chamber lid seal degradation

## Process Stages
- Etch
- Lithography

## Typical Corrective Actions
- Replace chamber O-rings (P/N format: 839-XXXX)
- Inspect/replace plasma confinement ring
- Calibrate EBR nozzle position (spec: ±0.5mm)
- Leak check after replacement (<1 mTorr/min)
- Run 5-wafer seasoning sequence before production

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| O-ring life | 2500 RF-hours max |
| Leak rate | <1 mTorr/min |
| Edge/center etch uniformity | <5% |
| EBR nozzle position | ±0.5mm |
| Seasoning wafers | 5 minimum |

## Correlation Clues to Include in Samples
- "Tool at 2,800 RF-hours" → exceeds O-ring limit
- "After PM, seal not replaced" → maintenance gap
- "Edge etch rate 12% higher" → quantitative evidence
- "Multiple lots affected on same tool" → systematic tool issue

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Edge-Ring**

KNOWLEDGE BASE:
- Visual Pattern: Defects forming ring pattern at wafer edge, 5-15mm from periphery
- Known Root Causes:
  1. Aged O-ring seal in etch chamber (typical life: 2500 RF-hours)
  2. Plasma confinement ring erosion affecting edge plasma density
  3. Edge bead removal (EBR) nozzle misalignment or clog
  4. Clamp ring contamination or wear
  5. Chamber lid seal degradation
- Process Stages: Etch, Lithography
- Corrective Actions:
  1. Replace chamber O-rings (P/N format: 839-XXXX)
  2. Inspect/replace plasma confinement ring
  3. Calibrate EBR nozzle position (spec: ±0.5mm)
  4. Leak check after replacement (<1 mTorr/min)
  5. Run 5-wafer seasoning sequence

TOOL VENDORS: LAM, AMAT, TEL
TOOL ID FORMAT: ETCH-LAM-07, ETCH-AMAT-03, LITH-TEL-12

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Defect observation (2-3 sentences)
- Lot ID (W2024-XXXX), wafer count, yield loss %
- Tool ID, recipe name
- Recent maintenance/RF-hour info
- Correlation clues where relevant

**OUTPUT** (Expert analysis):
- Root cause with specific mechanism
- Quantitative specs (RF-hours, leak rates, uniformity %)
- Part numbers (P/N: XXX-XXXX)
- Numbered corrective actions with validation
- Prevention measures with FDC recommendations
- Severity and yield impact

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}

Generate 25 unique, varied samples.
```
