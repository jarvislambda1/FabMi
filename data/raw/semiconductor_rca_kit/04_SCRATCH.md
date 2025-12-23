# Defect Type: SCRATCH

## Visual Pattern
Linear marks across wafer surface, various orientations and lengths (typically 15-50mm)

## Known Root Causes
- FOUP/cassette slot burr or damage (check SPECIFIC slots if pattern exists)
- Robot end effector wear or contamination
- CMP pad glazing or embedded particle
- Wafer handler misalignment
- Particle trapped during transfer

## Process Stages
- Handling
- CMP
- Transport

## Typical Corrective Actions
- Correlate affected wafers to FOUP slot positions (CRITICAL)
- Inspect specific cassette slots with magnification
- Replace/clean robot end effector
- Check CMP pad condition and diamond dresser
- Quarantine suspect FOUP immediately

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| End effector replacement | Every 50K wafer moves |
| CMP pad life | 500-1000 wafers (vendor spec) |
| FOUP inspection | Every 100 uses |
| Scratch detection limit | >0.5µm width, >5mm length |

## ⚠️ CRITICAL: Correlation Clues (KEY DIFFERENTIATOR)

This defect type is the **best for demonstrating correlation reasoning** - the pattern of WHICH wafers are affected reveals the root cause:

| Affected Wafers | Root Cause |
|-----------------|------------|
| 1, 3, 5, 7... (odd slots) | Odd-numbered FOUP slots damaged |
| 2, 4, 6, 8... (even slots) | Even-numbered FOUP slots damaged |
| 3, 5, 7, 9 (specific slots) | Those specific slots have burrs |
| All wafers, same orientation | Robot end effector issue |
| Random wafers, same lot | CMP particle issue |
| Only lot X, not lot Y (same robot) | FOUP-specific, not robot |

### Example Reasoning:
> "Scratches on wafers 3, 5, 7, 9 in FOUP-2847, but no scratches on wafers in FOUP-2848 processed with same robot"
> 
> **Conclusion**: FOUP-2847 slots 3, 5, 7, 9 are damaged. Robot is NOT the issue (different FOUP, same robot = no scratches).

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Scratch**

KNOWLEDGE BASE:
- Visual Pattern: Linear marks across wafer surface, various orientations (15-50mm length)
- Known Root Causes:
  1. FOUP/cassette slot burr or damage (check SPECIFIC slots)
  2. Robot end effector wear or contamination
  3. CMP pad glazing or embedded particle
  4. Wafer handler misalignment
  5. Particle trapped during transfer
- Process Stages: Handling, CMP, Transport
- Corrective Actions:
  1. Correlate affected wafers to FOUP slot positions
  2. Inspect specific cassette slots with magnification
  3. Replace/clean robot end effector (P/N: 412-XXXX)
  4. Check CMP pad condition and diamond dresser
  5. Quarantine suspect FOUP immediately

CRITICAL - INCLUDE CORRELATION CLUES:
- At least 10 samples must have wafer slot patterns (e.g., "wafers 3,5,7,9 affected")
- At least 5 samples must have FOUP comparison (e.g., "FOUP-A affected, FOUP-B same robot not affected")
- Show the REASONING in the output that connects the pattern to root cause

TOOL VENDORS: Various handlers, CMP tools
TOOL ID FORMAT: HANDLER-BROOKS-02, CMP-AMAT-05, FOUP-XXXX

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Scratch description (orientation, length, location)
- WHICH SPECIFIC WAFERS affected (slot numbers!)
- Lot ID, FOUP ID used
- Robot/handler ID
- CMP or handling history

**OUTPUT** (Expert analysis):
- Root cause with CORRELATION REASONING
- Why the pattern indicates this specific cause
- Rule out alternatives (e.g., "robot ruled out because...")
- Numbered corrective actions
- Prevention measures
- Severity and yield impact

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}

Generate 25 unique samples with strong correlation reasoning.
```
