---
library_name: peft
license: other
base_model: baidu/ERNIE-4.5-0.3B-PT
tags:
- base_model:adapter:baidu/ERNIE-4.5-0.3B-PT
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: ernie_semi_rca
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ernie_semi_rca

This model is a fine-tuned version of [baidu/ERNIE-4.5-0.3B-PT](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT) on the semi_rca dataset.
It achieves the following results on the evaluation set:
- Loss: 1.5169

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.677         | 1.1770 | 100  | 1.6367          |
| 1.5008        | 2.3540 | 200  | 1.5224          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.56.1
- Pytorch 2.9.1+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1