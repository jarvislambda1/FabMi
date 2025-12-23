# Defect Type: DONUT

## Visual Pattern
Ring-shaped defect pattern with clear center, like a donut

## Known Root Causes
- Tungsten loss (W-loss) in CVD process
- Edge ring erosion in plasma chamber
- Chamber seasoning film peeling
- ESC (Electrostatic Chuck) backside He leak
- RF coupling non-uniformity

## Process Stages
- CVD
- Etch
- PVD

## Typical Corrective Actions
- Replace/recondition edge ring
- Perform chamber wet clean + conditioning
- Check ESC for He leak (spec: <2 sccm)
- Verify deposition uniformity with monitor wafer
- Replace chamber liner if >5000 RF-hours

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| Edge ring life | 3000-5000 RF-hours |
| He backside leak | <2 sccm |
| Deposition uniformity | <3% |
| Chamber liner life | 5000 RF-hours |
| W-loss threshold | <5% film loss |

## Correlation Clues to Include in Samples
- "Edge ring at 4,500 RF-hours" → approaching end of life
- "He flow higher than normal" → ESC leak indication
- "Ring radius consistent across lot" → systematic chamber issue
- "After extended idle" → seasoning film degradation

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Donut**

KNOWLEDGE BASE:
- Visual Pattern: Ring-shaped defect pattern with clear center, like a donut
- Known Root Causes:
  1. Tungsten loss (W-loss) in CVD process
  2. Edge ring erosion in plasma chamber
  3. Chamber seasoning film peeling
  4. ESC (Electrostatic Chuck) backside He leak
  5. RF coupling non-uniformity
- Process Stages: CVD, Etch, PVD
- Corrective Actions:
  1. Replace/recondition edge ring (P/N format: 715-XXXX)
  2. Perform chamber wet clean + conditioning
  3. Check ESC for He leak (spec: <2 sccm)
  4. Verify deposition uniformity with monitor wafer
  5. Replace chamber liner if >5000 RF-hours

TOOL VENDORS: AMAT, LAM, TEL
TOOL ID FORMAT: CVD-AMAT-02, ETCH-LAM-09, PVD-AMAT-06

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Defect observation with donut-specific details (ring diameter, clear center)
- Lot ID (W2024-XXXX), wafer count, yield loss %
- Tool ID, recipe name
- RF-hours, He flow readings if relevant
- Chamber/edge ring maintenance history

**OUTPUT** (Expert analysis):
- Root cause with specific mechanism
- Quantitative specs (RF-hours, He leak rate, uniformity %)
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
