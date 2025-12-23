# Defect Type: EDGE-LOC (Edge Localized)

## Visual Pattern
Localized defects at specific edge position (not full ring) - defects at one "clock position"

## Known Root Causes
- Notch/flat alignment sensor miscalibration
- Single-point contamination on chuck edge
- Robot arm contact at specific position
- Localized EBR failure
- Handler lift pin damage

## Process Stages
- Handling
- Lithography

## Typical Corrective Actions
- Check notch finder calibration (spec: ±0.1°)
- Inspect chuck edge at defect clock position
- Review robot teach points
- Check specific EBR nozzle if at resist edge
- Verify handler lift pins condition

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| Notch alignment | ±0.1° |
| Chuck edge inspection | Every PM |
| Robot teach point verification | Monthly |
| EBR nozzle position | ±0.3mm |
| Lift pin height | Per vendor spec |

## Correlation Clues to Include in Samples

| Pattern | Root Cause |
|---------|------------|
| Defects at 3 o'clock on all wafers | Chuck contamination at that position |
| Defects at notch position | Notch finder contact issue |
| Defects rotate with wafer orientation | Pre-existing before alignment |
| Defects fixed position regardless of notch | Post-alignment chuck issue |
| Only on certain recipe | Process-specific edge effect |

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Edge-Loc (Edge Localized)**

KNOWLEDGE BASE:
- Visual Pattern: Localized defects at specific edge position (clock position), not full ring
- Known Root Causes:
  1. Notch/flat alignment sensor miscalibration
  2. Single-point contamination on chuck edge
  3. Robot arm contact at specific position
  4. Localized EBR failure
  5. Handler lift pin damage
- Process Stages: Handling, Lithography
- Corrective Actions:
  1. Check notch finder calibration (spec: ±0.1°)
  2. Inspect chuck edge at defect clock position
  3. Review robot teach points
  4. Check specific EBR nozzle position
  5. Verify handler lift pins condition

TOOL VENDORS: TEL, ASML, Brooks, Yaskawa
TOOL ID FORMAT: LITH-TEL-05, HANDLER-BROOKS-03, COAT-TEL-08

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Defect location as clock position (e.g., "3 o'clock", "near notch")
- Whether position is fixed or rotates with wafer
- Lot ID, wafer count, yield loss %
- Tool ID, recipe name
- Alignment/handling history

**OUTPUT** (Expert analysis):
- Root cause with position correlation reasoning
- Whether defect is pre or post alignment
- Numbered corrective actions with validation
- Prevention measures
- Severity and yield impact

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}

Generate 25 unique, varied samples.
```
