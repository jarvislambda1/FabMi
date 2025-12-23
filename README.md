# FabMi: Semiconductor Defect Root Cause Analysis with ERNIE

<!-- Banner image placeholder - add your own branded image -->
<!-- <p align="center">
  <img src="docs/assets/fabmi_banner.png" alt="FabMi Banner" width="800">
</p> -->

**Fine-tuned ERNIE-4.5-0.3B for automated semiconductor wafer defect diagnosis, achieving 607% improvement in ROUGE-L over zero-shot baseline while being 70x smaller.**

[![ERNIE](https://img.shields.io/badge/Model-ERNIE--4.5--0.3B-blue)](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT)
[![LLaMA-Factory](https://img.shields.io/badge/Framework-LLaMA--Factory-green)](https://github.com/hiyouga/LLaMA-Factory)
[![License](https://img.shields.io/badge/License-Apache%202.0-red)](LICENSE)

---

## The Problem & Market Opportunity

### The $600B Industry's Hidden Bottleneck

The global semiconductor industry is projected to exceed **$1 trillion by 2030**, yet fabs face a critical knowledge bottleneck: root cause analysis still depends on senior engineers with 15+ years of experience.

When a defect pattern appears on a wafer map, existing AI systems can classify *what* the defect is (scratch, edge-ring, center) with 96%+ accuracy—but they **cannot explain *why* it happened or *how* to fix it**.

That critical reasoning step still requires a human expert to:
- Correlate patterns (which specific wafer slots are affected? what changed after PM?)
- Reference tribal knowledge
- Prescribe corrective actions

With the average fab engineer age rising and fewer graduates entering the field, this expertise gap is becoming existential. **A single day of yield loss at a leading-edge fab can cost $5-10 million.**

### The Opportunity: From Classification to Reasoning

Current defect detection is a **$4.2B market** dominated by KLA, Applied Materials, and Onto Innovation—but it stops at detection. The root cause analysis and corrective action step remains manual, taking hours or days while defective wafers continue through the line.

**FabMi bridges this gap**: by fine-tuning an LLM on expert-level troubleshooting patterns, we transform defect classification outputs into actionable RCA reports with:
- Specific part numbers
- Quantitative thresholds
- Validation criteria

This isn't replacing vision systems—it's **completing the loop**. The model learns correlation reasoning that base LLMs fail at entirely:

> *"Wafers 3,5,7,9 affected means FOUP slots, not robot arm."*

This is the difference between generic advice and a **$50K/hour senior process engineer's insight delivered in seconds**.

---

## Key Results

### Performance Comparison

| Metric | Baseline (21B zero-shot) | FabMi (0.3B) | Improvement |
|--------|--------------------------|----------------|-------------|
| **ROUGE-L** | 0.063 | 0.446 | **+607%** |
| **ROUGE-1** | 0.101 | 0.561 | **+453%** |
| **ROUGE-2** | 0.028 | 0.282 | **+899%** |
| **Severity Accuracy** | 20.0% | 60.0% | **+200%** |
| **Structure Score** | 35.5% | 88.0% | **+148%** |
| **Model Size** | 21B | 0.3B | **70x smaller** |

### Training Progression

| Metric | 3-epoch | 10-epoch | Improvement |
|--------|---------|----------|-------------|
| ROUGE-L | 0.404 | 0.446 | +10.4% |
| Severity Acc | 50.3% | 60.0% | +19.3% |
| Structure | 89.7% | 88.0% | -1.9% |

---

## How It Works

### 1. Data Generation Pipeline

We created **1,511 synthetic RCA training samples** covering 9 defect patterns using multiple state-of-the-art LLMs:

| Defect Type | Samples | Description |
|-------------|---------|-------------|
| Edge-Ring | 242 | O-ring seal degradation patterns |
| Donut | 186 | Center-excluded ring patterns |
| Center | 180 | Center-concentrated defects |
| Scratch | 174 | Linear mechanical damage |
| Edge-LOC | 162 | Edge-localized clusters |
| Random | 158 | Random particle distribution |
| LOC | 152 | Localized defect clusters |
| Near-Full | 147 | Catastrophic failures (>80%) |
| None | 110 | Normal baseline (no defect) |

**Data Generation Sources:**
- Claude Opus 4.5
- Gemini 3 Pro
- GPT-4.5
- Grok

### 2. Training Approach

- **Base Model**: `baidu/ERNIE-4.5-0.3B-PT`
- **Method**: LoRA fine-tuning (rank=16, alpha=32)
- **Framework**: LLaMA-Factory
- **Training**: 10 epochs, cosine LR schedule
- **Data Split**: 90% train (1,356) / 10% test (155)

### 3. Output Format

**Input:**
```
Defect Observation: Edge-ring pattern detected on lot W2024-3847.
15% yield loss concentrated at wafer edge (>140mm radius).
All 25 wafers in lot affected uniformly.

Process Context: Wafers processed through ETCH-LAM-07.
Chamber at 2,847 RF-hours. Last PM was 3 weeks ago.
O-ring inspection not performed during PM.
```

**Output:**
```markdown
## Root Cause Analysis

**Primary Cause**: O-ring seal degradation in etch chamber
- RF-hours (2,847) approaching 3,000-hour O-ring life limit
- Edge-ring pattern indicates gas leakage at chamber seal
- Uniform lot impact confirms chamber-level issue

## Corrective Actions

1. **Immediate**: Tool down ETCH-LAM-07
2. **Inspection**: Check main chamber O-ring for degradation
3. **Replace**: O-ring P/N 839-0234 (Viton, ID 350mm)
4. **Verify**: Leak rate <1 mTorr/min after replacement
5. **Qualify**: Run 5-wafer particle qualification

**Severity**: Critical
**Yield Impact**: 15%
```

---

## Repository Structure

```
FabMi/
├── README.md
├── data/
│   ├── raw/semiconductor_rca_kit/    # Defect type templates
│   ├── generated/                     # LLM-generated samples
│   │   ├── opus_4.5/
│   │   ├── gemini_3_pro/
│   │   ├── gpt_4.5/
│   │   └── grok/
│   ├── merged/                        # Consolidated dataset
│   └── splits/                        # Train/test splits
├── configs/
│   ├── ernie_semi_rca_3ep.yaml       # 3-epoch config
│   ├── ernie_semi_rca_10ep.yaml      # 10-epoch config
│   └── merge_config.yaml              # LoRA merge config
├── models/
│   ├── ernie_semi_rca_3ep/           # 3-epoch LoRA adapter (~12MB)
│   ├── ernie_semi_rca_10ep/          # 10-epoch LoRA adapter (~24MB)
│   └── ernie_semi_rca_merged/        # Merged model (optional)
├── eval/
│   ├── eval_baseline.py              # Zero-shot evaluation
│   ├── eval_finetuned.py             # Fine-tuned evaluation
│   └── results/                       # Evaluation JSONs
├── scripts/
│   ├── data_generation/
│   ├── training/
│   └── serving/
├── demo/
│   └── app.py                         # Gradio demo
└── docs/
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/FabMi.git
cd FabMi
pip install -r requirements.txt
```

### Inference

**Option 1: Using LoRA adapter (recommended)**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "baidu/ERNIE-4.5-0.3B-PT",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "./models/ernie_semi_rca_10ep")
tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-0.3B-PT", trust_remote_code=True)

prompt = """Analyze this semiconductor wafer defect and provide root cause analysis.

Defect Observation: Center-concentrated defects on lot W2024-5521.
20% yield loss within 45mm radius of wafer center.

Process Context: CVD-AMAT-03, showerhead at 3,200 RF-hours.
Center zone showing reduced deposition rate."""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Option 2: Using merged model (after running merge)**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/ernie_semi_rca_merged", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./outputs/ernie_semi_rca_merged", trust_remote_code=True)

# Same inference code as above...
```

**Option 3: Via OpenAI-compatible API**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=512
)
print(response.choices[0].message.content)
```

---

## Reproducing Results

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/FabMi.git
cd FabMi

# Install dependencies
pip install -r requirements.txt

# Install LLaMA-Factory
pip install llamafactory
```

### Step 1: Register Dataset with LLaMA-Factory

Add the dataset to LLaMA-Factory's dataset registry:

```bash
# Find your LLaMA-Factory installation
LLAMA_FACTORY_PATH=$(python -c "import llamafactory; print(llamafactory.__path__[0])")

# Add dataset entry to dataset_info.json
cat >> "$LLAMA_FACTORY_PATH/data/dataset_info.json" << 'EOF'
{
  "semi_rca": {
    "file_name": "/path/to/FabMi/data/splits/train.json",
    "formatting": "alpaca"
  },
  "semi_rca_test": {
    "file_name": "/path/to/FabMi/data/splits/test.json",
    "formatting": "alpaca"
  }
}
EOF
```

> **Note:** Update `/path/to/FabMi` with your actual path.

### Step 2: Train the Model

```bash
# 10-epoch training (recommended)
llamafactory-cli train configs/ernie_semi_rca_10ep.yaml

# Or 3-epoch for faster iteration
llamafactory-cli train configs/ernie_semi_rca_3ep.yaml
```

**Training takes ~30 minutes on a single GPU (RTX 3090/4090).**

Output: `./outputs/ernie_semi_rca_10ep/` containing the LoRA adapter (~24MB)

### Step 3: Merge LoRA with Base Model (Optional)

```bash
# Merge LoRA adapter into base model for faster inference
llamafactory-cli export configs/merge_config.yaml
```

Output: `./outputs/ernie_semi_rca_merged/` (~700MB full model)

> **Note:** Merging is optional. You can serve the LoRA adapter directly.

### Step 4: Serve the Model

**Option A: Serve merged model**
```bash
llamafactory-cli api \
    --model_name_or_path ./outputs/ernie_semi_rca_merged \
    --template ernie_nothink
```

**Option B: Serve with LoRA adapter (no merge needed)**
```bash
llamafactory-cli api \
    --model_name_or_path baidu/ERNIE-4.5-0.3B-PT \
    --adapter_name_or_path ./models/ernie_semi_rca_10ep \
    --template ernie_nothink
```

Server starts at `http://localhost:8000` with OpenAI-compatible API.

### Step 5: Run the Demo

```bash
# Set environment variables
export FABMI_API_URL="http://localhost:8000/v1"
export FABMI_MODEL_NAME="gpt-3.5-turbo"  # LLaMA-Factory default alias

# Start Gradio demo
python demo/app.py
```

Open `http://localhost:7860` in your browser.

---

## Evaluation

Run evaluation on test set:

```bash
python eval/eval_finetuned_parallel.py --model ./models/ernie_semi_rca_merged --workers 8
```

### Results by Defect Type

| Defect Type | ROUGE-L | Severity Acc | Structure |
|-------------|---------|--------------|-----------|
| None | 0.598 | 72.7% | 97.4% |
| Edge-LOC | 0.481 | 64.7% | 92.4% |
| Donut | 0.441 | 57.9% | 90.2% |
| Near-Full | 0.436 | 73.3% | 85.7% |
| Edge-Ring | 0.428 | 56.0% | 88.6% |
| Random | 0.422 | 50.0% | 83.9% |
| LOC | 0.409 | 56.3% | 83.0% |
| Center | 0.387 | 55.6% | 87.3% |
| Scratch | 0.387 | 55.6% | 84.4% |

---

## Demo

Try the interactive demo:

```bash
python demo/app.py
```

<!-- Screenshot placeholder - add your own demo screenshot -->
<!-- <p align="center">
  <img src="docs/assets/demo_screenshot.png" alt="Gradio Demo" width="800">
</p> -->

---

## Citation

```bibtex
@software{fabmi2024,
  title={FabMi: Semiconductor Defect Root Cause Analysis with ERNIE},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/FabMi}
}
```

---

## Acknowledgments

- [Baidu ERNIE](https://github.com/PaddlePaddle/ERNIE) for the base model
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the fine-tuning framework
- [ERNIE AI Developer Challenge](https://baiduernieai.devpost.com/) hackathon

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
# FabMi
