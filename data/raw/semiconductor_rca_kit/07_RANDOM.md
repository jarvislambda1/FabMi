# Defect Type: RANDOM

## Visual Pattern
Randomly distributed defects across entire wafer, no systematic pattern

## Known Root Causes
- Particle contamination in cleanroom/mini-environment
- Process chemical degradation or particles
- Cross-contamination between lots
- AMC (Airborne Molecular Contamination)
- HEPA/ULPA filter degradation

## Process Stages
- Environment
- Wet Clean
- Multiple (can occur anywhere)

## Typical Corrective Actions
- Check cleanroom particle counts (spec: Class 1 or ISO 3)
- Sample and test process chemicals
- Review lot segregation procedures
- Check chemical filter lifetime
- Verify HEPA filter integrity

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| Cleanroom particles | <1 particle/ft³ @ 0.1µm (Class 1) |
| Chemical filters | Per vendor lifetime |
| HEPA efficiency | >99.99% @ 0.3µm |
| AMC levels | <1 ppb for critical species |
| DI water resistivity | >18 MΩ·cm |

## Correlation Clues to Include in Samples

| Pattern | Root Cause |
|---------|------------|
| All tools affected | Cleanroom/environment issue |
| Single tool, random pattern | Tool-specific contamination |
| After chemical change | Chemical quality issue |
| After filter replacement | Filter installation issue |
| Increasing trend over time | Gradual contamination buildup |
| Specific shift affected | Operator procedure issue |

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Random**

KNOWLEDGE BASE:
- Visual Pattern: Randomly distributed defects across entire wafer, no systematic pattern
- Known Root Causes:
  1. Particle contamination in cleanroom/mini-environment
  2. Process chemical degradation or particles
  3. Cross-contamination between lots
  4. AMC (Airborne Molecular Contamination)
  5. HEPA/ULPA filter degradation
- Process Stages: Environment, Wet Clean, Multiple
- Corrective Actions:
  1. Check cleanroom particle counts (spec: Class 1/ISO 3)
  2. Sample and test process chemicals
  3. Review lot segregation procedures
  4. Check chemical filter lifetime
  5. Verify HEPA filter integrity

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Random defect distribution description
- Whether multiple tools/areas affected
- Lot ID, wafer count, yield loss %
- Recent chemical/filter changes
- Particle count trends if available

**OUTPUT** (Expert analysis):
- Root cause with environmental correlation
- How to distinguish tool-specific vs. fab-wide issue
- Quantitative specs (particle counts, AMC levels)
- Numbered corrective actions with validation
- Prevention measures with monitoring recommendations
- Severity and yield impact

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}

Generate 25 unique, varied samples.
```
