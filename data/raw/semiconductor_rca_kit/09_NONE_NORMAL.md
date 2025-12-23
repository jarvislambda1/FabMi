# Defect Type: NONE (Normal)

## Visual Pattern
No systematic defect pattern, normal baseline defectivity within spec

## Purpose in Training Data
- Provides contrast to defective samples
- Trains model to recognize when NO action is needed
- Prevents false positives / over-diagnosis

## Typical Scenario
- Process running within specification
- Normal random defectivity below threshold
- Yield within expected range

## Sample Count
Generate only **10-15 samples** for this type (much fewer than defect types)

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 10 unique training samples for: **None (Normal/No Defect)**

These samples show NORMAL operation where no root cause analysis is needed.

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Normal wafer inspection results
- Lot ID, wafer count
- Yield within expected range (e.g., 92-98%)
- Defect density below threshold
- Tool ID, recipe (all running normally)

**OUTPUT** (Expert analysis):
- Confirmation that no systematic defect pattern exists
- Defectivity is within baseline specification
- No corrective action required
- Continue standard monitoring
- Severity: None, Yield Impact: Within spec

EXAMPLES OF NORMAL SCENARIOS:
- "Lot W2024-5521 processed through CVD-AMAT-03. Yield at 94.2%, within 92-96% target. Defect density 0.08/cm², below 0.15/cm² threshold. No systematic pattern observed."
- "Random inspection of 5 wafers from lot W2024-6632. All wafers show normal baseline defectivity. No action required."

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Analysis\n\n**Finding**: No systematic defect pattern detected..."
}

Generate 10 unique normal/no-defect samples.
```
