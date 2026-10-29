---
license: bigscience-bloom-rail-1.0
tags:
- generated_from_trainer
datasets:
- c4
model-index:
- name: bloom-7b1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bloom-7b1

This model is a fine-tuned version of [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) on the c4 en dataset.
It achieves the following results on the evaluation set:
- eval_loss: 2.9793
- eval_accuracy: 0.4151
- eval_runtime: 3789.1944
- eval_samples_per_second: 0.528
- eval_steps_per_second: 0.528
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
