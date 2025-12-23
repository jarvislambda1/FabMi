# Synthetic Data Generation Plan: Semiconductor RCA

## Objective
Generate 530+ high-quality training samples for Semiconductor Root Cause Analysis (RCA) using direct AI generation based on the provided prompt templates.

## Strategy: Direct AI Synthesis
Instead of code scripts, we will leverage the AI's capability to act as the domain expert (Senior Process Engineer) and directly synthesize the training data in JSON format.

## Implementation Steps

### 1. Defect-Specific Generation
For each defect type defined in the kit, we will generate a JSON file containing unique samples.
*   **Source Material**: Use the `KNOWLEDGE BASE` from files like `01_EDGE_RING.md`, `02_CENTER.md`, etc.
*   **Format**: Alpaca-style JSON (`instruction`, `input`, `output`).
*   **Target Directory**: `semiconductor_rca_kit/data/attempt2/`

### 2. Validation & Correlation
Ensure each batch includes the "WOW Factor" correlation reasoning:
*   **Scratch**: Correlate wafer slots to FOUP slots (e.g., "Odd slots only" -> "FOUP A side").
*   **Edge Ring**: Correlate RF hours to O-ring life (e.g., ">2500 hrs").

### 3. Output Files
We will create the following files directly in `data/attempt2/`:
*   `data_01_edge_ring.json`
*   `data_02_center.json`
*   `data_04_scratch.json`
*   ... (and so on)

### 4. Merging
Finally, combine all individual JSON files into the master `semiconductor_rca_data.json`.

## Data Dictionary (Mental Model)
*   **Vendors**: LAM, AMAT, TEL, KLA, ASML.
*   **Tools**: ETCH-LAM-07, CVD-AMAT-03, CMP-EBRA-01.
*   **Parts**: O-rings (839-XXXX), Focus Rings (715-XXXX).
*   **Specs**: Uniformity <2%, Leak Rate <1 mTorr/min.
