# Defect Type: NEAR-FULL

## Visual Pattern
Defects covering >80% of wafer surface - catastrophic failure

## Known Root Causes
- Complete process recipe failure
- RF power supply failure during process
- Gas delivery system failure
- Massive contamination event
- Vacuum breach during process

## Process Stages
- Critical Failure
- Can occur at any process step

## Typical Corrective Actions
- IMMEDIATE tool down - do not process more wafers
- Quarantine all wafers in chamber at failure time
- Review FDC data at exact failure timestamp
- Check all utilities (RF, gas, vacuum, cooling)
- Escalate to root cause team

## Quantitative Specs
| Parameter | Notes |
|-----------|-------|
| Severity | ALWAYS Critical |
| Yield impact | 80-100% |
| Response | Immediate tool down |
| Investigation | Full failure analysis required |

## ⚠️ This is Always Critical

Near-full defects indicate catastrophic process failure. The focus is on:
1. Immediate containment
2. Identifying exact failure point
3. Preventing recurrence

## Correlation Clues to Include in Samples

| Pattern | Root Cause |
|---------|------------|
| Mid-lot failure | Equipment malfunction during process |
| First wafer of lot | Recipe load error or chamber not ready |
| After extended idle (>4 hrs) | Chamber conditioning issue |
| Multiple tools same time | Facility issue (gas, power) |
| Single tool, sudden onset | Tool-specific failure |
| Gradual degradation over lot | Progressive equipment failure |

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 20 unique defect root cause analysis training samples for: **Near-Full**

KNOWLEDGE BASE:
- Visual Pattern: Defects covering >80% of wafer surface - catastrophic failure
- Known Root Causes:
  1. Complete process recipe failure
  2. RF power supply failure during process
  3. Gas delivery system failure
  4. Massive contamination event
  5. Vacuum breach during process
- Process Stages: Critical Failure (any step)
- Corrective Actions:
  1. IMMEDIATE tool down
  2. Quarantine all wafers in chamber
  3. Review FDC data at failure timestamp
  4. Check all utilities (RF, gas, vacuum, cooling)
  5. Escalate to root cause team

CRITICAL: All samples should be Critical severity with 80-100% yield impact

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Catastrophic defect description
- When in lot the failure occurred (first wafer, mid-lot, etc.)
- Lot ID, wafers affected
- Tool ID, recipe name
- Any utility or equipment alarms

**OUTPUT** (Expert analysis):
- Root cause with failure mechanism
- Timeline analysis (when did failure occur)
- Immediate containment actions
- Investigation steps with FDC review
- Prevention measures
- Severity: Critical, Yield Impact: 80-100%

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}

Generate 20 unique samples showing various catastrophic failure scenarios.
```
