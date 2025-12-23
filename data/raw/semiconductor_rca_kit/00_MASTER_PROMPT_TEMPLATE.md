# Master Prompt Template for Semiconductor RCA Data Generation

## How to Use

1. Copy this master prompt
2. Replace `{VARIABLES}` with content from each defect type file
3. Ask Claude/GPT to generate 25 samples
4. Repeat for all 9 defect types
5. Combine all JSON outputs into single training file

---

## Master Prompt

```
You are a senior semiconductor process engineer with 15 years of fab experience at TSMC/Samsung/Intel.

Generate 25 unique defect root cause analysis training samples for: **{DEFECT_TYPE}**

KNOWLEDGE BASE FOR THIS DEFECT:
- Visual Pattern: {VISUAL}
- Known Root Causes: {ROOT_CAUSES}
- Process Stages: {STAGES}
- Typical Corrective Actions: {ACTIONS}

TOOL VENDORS TO USE: LAM, AMAT, TEL, ASML, KLA, Screen, Tokyo Electron
TOOL ID FORMAT: {PROCESS}-{VENDOR}-{NUMBER} (e.g., ETCH-LAM-07, CVD-AMAT-03)
PART NUMBER FORMAT: XXX-XXXX (e.g., 839-0127)

FOR EACH SAMPLE, GENERATE:

**INPUT** (Engineer's observation):
- Defect observation (2-3 sentences with specific details)
- Lot ID (format: W2024-XXXX), wafer count, yield loss %
- Tool ID, recipe name
- Recent process history / maintenance info
- Include CORRELATION CLUES where relevant (e.g., "wafers 2,4,6 affected", "last 3 lots", "after PM")

**OUTPUT** (Expert analysis):
- Root cause with specific technical mechanism
- Quantitative specs (RF-hours, uniformity %, flow rates, leak rates)
- Specific part numbers where applicable
- Numbered corrective actions with validation steps
- Prevention measures with FDC/monitoring recommendations
- Severity (Critical/Major/Minor) and yield impact %

CRITICAL REQUIREMENTS:
1. Each scenario must be UNIQUE - vary tools, lots, specific values
2. Include CORRELATION REASONING (slot patterns, time patterns, tool patterns)
3. Use QUANTITATIVE THRESHOLDS (>2500 RF-hours, <1 mTorr/min, <2% variation)
4. Add VALIDATION CRITERIA (how to confirm fix worked)

Output as JSON array with format:
{
  "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
  "input": "Defect Observation: ... Process Context: ...",
  "output": "## Root Cause Analysis\n\n**Primary Cause:**..."
}
```

---

## Output JSON Format (Alpaca Style)

```json
[
  {
    "instruction": "Analyze this semiconductor wafer defect and provide root cause analysis with corrective actions.",
    "input": "Defect Observation: Edge-ring pattern detected on lot W2024-3821. Wafer maps show 18% yield loss concentrated in a ring pattern 5-10mm from wafer edge. Pattern consistent across 6 wafers. Defects appear after plasma etch step.\n\nProcess Context: Wafers processed through ETCH-LAM-07. Recipe: POLY_GATE_STD. Tool had preventive maintenance 2,800 RF-hours ago. Previous lot showed normal yield.",
    "output": "## Root Cause Analysis\n\n**Primary Cause**: O-ring seal degradation in etch chamber\n- Tool ETCH-LAM-07 at 2,800 RF-hours exceeds O-ring life limit of 2,500 RF-hours\n- Degraded seal causing plasma leak at chamber edge\n- Edge etch rate 12% higher than center (spec: <5%)\n\n**Contributing Factors**:\n- PM 2,800 hours ago did not include O-ring replacement\n- No FDC alarm configured for edge/center uniformity\n\n## Corrective Actions\n\n1. **Immediate**: Tool down ETCH-LAM-07\n2. **Replace O-ring**: P/N 839-0127, torque to 15 ft-lbs\n3. **Leak check**: Verify <1 mTorr/min leak rate\n4. **Seasoning**: Run 5-wafer conditioning sequence\n5. **Validation**: Process qual wafer, verify edge/center uniformity <5%\n\n## Prevention Measures\n\n- Add O-ring RF-hour tracking to PM system (replace at 2,500 hrs)\n- Configure FDC alarm for edge/center etch ratio >8%\n- Update PM checklist to include leak verification\n\n**Severity**: Critical\n**Yield Impact**: 18%"
  }
]
```

---

## Generation Checklist

For each defect type, ensure samples include:

- [ ] Specific tool IDs (ETCH-LAM-07, CVD-AMAT-03)
- [ ] Part numbers (P/N: 839-0127)
- [ ] Quantitative specs (<2%, >2500 RF-hrs, <1 mTorr/min)
- [ ] Correlation clues (wafer patterns, time patterns)
- [ ] Validation criteria
- [ ] FDC/monitoring recommendations
- [ ] Severity and yield impact

---

## Target Sample Count

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
