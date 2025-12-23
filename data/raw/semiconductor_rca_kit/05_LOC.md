# Defect Type: LOC (Localized Cluster)

## Visual Pattern
Localized cluster of defects at specific wafer location, not following edge or center pattern

## Known Root Causes
- Particle fall-on from chamber ceiling/showerhead
- Localized mask/reticle defect
- Resist bubble during coating
- Foreign material on pellicle
- Single-point contamination on chuck

## Process Stages
- Lithography
- Deposition
- Environment

## Typical Corrective Actions
- Map defect location across multiple wafers (same XY = systematic)
- Inspect reticle/pellicle for contamination
- Check chamber ceiling with particle wafer
- Review resist dispense for bubbles
- Clean specific chuck location if defect fixed in XY

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| Reticle inspection | Every 1000 exposures |
| Particle wafer threshold | <0.1 particles/cm² (>0.1µm) |
| Pellicle replacement | Every 6 months or per spec |
| Chuck cleaning | Every PM or when defect detected |

## Correlation Clues to Include in Samples

| Pattern | Root Cause |
|---------|------------|
| Same XY on all wafers | Reticle defect or chuck contamination |
| Random XY, same tool | Chamber particle fall-on |
| Only after specific recipe | Process-induced particle |
| Appears mid-lot, persists | Sudden contamination event |
| Only on specific reticle | Reticle/pellicle issue |

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Loc (Localized)**

KNOWLEDGE BASE:
- Visual Pattern: Localized cluster of defects at specific wafer location
- Known Root Causes:
  1. Particle fall-on from chamber ceiling/showerhead
  2. Localized mask/reticle defect
  3. Resist bubble during coating
  4. Foreign material on pellicle
  5. Single-point contamination on chuck
- Process Stages: Lithography, Deposition, Environment
- Corrective Actions:
  1. Map defect XY location across multiple wafers
  2. Inspect reticle/pellicle for contamination
  3. Check chamber ceiling with particle wafer
  4. Review resist dispense for bubbles
  5. Clean specific chuck location

TOOL VENDORS: ASML, Nikon, Canon (litho), AMAT, LAM (deposition)
TOOL ID FORMAT: LITH-ASML-03, CVD-AMAT-07, COAT-TEL-02

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Defect cluster location (XY coordinates or clock position)
- Whether location is consistent across wafers
- Lot ID, wafer count, yield loss %
- Tool ID, recipe, reticle ID if litho
- Recent process history

**OUTPUT** (Expert analysis):
- Root cause with XY correlation reasoning
- Why the pattern indicates specific cause
- Part numbers where applicable
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
