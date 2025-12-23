# Defect Type: CENTER

## Visual Pattern
Defects concentrated at wafer center, within 40-50mm radius

## Known Root Causes
- Non-uniform showerhead gas distribution (holes blocked/corroded)
- Chuck vacuum leak causing wafer bow at center
- CVD temperature gradient center-to-edge
- Spin coating acceleration profile incorrect
- Insufficient chamber seasoning after PM

## Process Stages
- CVD
- Lithography
- Thermal

## Typical Corrective Actions
- Disassemble and ultrasonically clean showerhead
- Check vacuum chuck for leaks (He leak check)
- Run 49-point thickness uniformity map
- Verify TEOS/gas flow center-edge variation <3%
- Run 5-10 seasoning wafers after PM

## Quantitative Specs
| Parameter | Spec |
|-----------|------|
| Thickness uniformity | <2% (1-sigma) |
| Gas flow variation | <3% center-to-edge |
| Chuck vacuum | >500 Torr |
| Seasoning wafers after PM | 5-10 minimum |
| Temperature uniformity | ±2°C across wafer |

## Correlation Clues to Include in Samples
- "Chamber opened 48 hours ago" → insufficient seasoning
- "Deposition rate trending down" → showerhead degradation
- "Center 8% thinner than edge" → quantitative evidence
- "After showerhead inspection" → PM-related

---

## Ready-to-Use Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience.

Generate 25 unique defect root cause analysis training samples for: **Center**

KNOWLEDGE BASE:
- Visual Pattern: Defects concentrated at wafer center, within 40-50mm radius
- Known Root Causes:
  1. Non-uniform showerhead gas distribution (holes blocked/corroded)
  2. Chuck vacuum leak causing wafer bow at center
  3. CVD temperature gradient center-to-edge
  4. Spin coating acceleration profile incorrect
  5. Insufficient chamber seasoning after PM
- Process Stages: CVD, Lithography, Thermal
- Corrective Actions:
  1. Disassemble and ultrasonically clean showerhead
  2. Check vacuum chuck for leaks (He leak check)
  3. Run 49-point thickness uniformity map
  4. Verify TEOS/gas flow center-edge variation <3%
  5. Run 5-10 seasoning wafers after PM

TOOL VENDORS: AMAT, LAM, TEL, ASM
TOOL ID FORMAT: CVD-AMAT-03, CVD-LAM-05, LITH-TEL-08

FOR EACH SAMPLE GENERATE:

**INPUT** (Engineer's observation):
- Defect observation with center-specific details
- Lot ID (W2024-XXXX), wafer count, yield loss %
- Tool ID, recipe name
- Recent PM/chamber history
- Thickness or uniformity measurements if relevant

**OUTPUT** (Expert analysis):
- Root cause with specific mechanism
- Quantitative specs (uniformity %, flow rates, vacuum levels)
- Part numbers where applicable
- Numbered corrective actions with validation criteria
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
