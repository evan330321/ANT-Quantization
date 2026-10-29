---
license: other
tags:
- generated_from_trainer
datasets:
- c4
model-index:
- name: opt-6.7b
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# opt-6.7b

This model is a fine-tuned version of [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) on the c4 en dataset.
It achieves the following results on the evaluation set:
- eval_loss: 4.9685
- eval_accuracy: 0.2042
- eval_runtime: 3856.4141
- eval_samples_per_second: 0.519
- eval_steps_per_second: 0.519
- step: 0

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.26.1
- Pytorch 2.1.0
- Datasets 2.15.0
- Tokenizers 0.13.3
