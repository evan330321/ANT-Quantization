#!/bin/bash
task_name=${1:-"squad"}
size=${2:-"base"}
gpu_num=${3:-0}
mode=${5:-"ant-int-flint"}
dataset_name=squad
if [ "$size" == "base" ] ; then
  path="ModelTC/bert-base-$task_name "
  batch_size=${4:-"64"}
fi
if [ "$size" == "large" ] ; then
  if [ "$task_name" == "squad" ] ; then
    path="bert-large-uncased-whole-word-masking-finetuned-squad"
  else
    path="deepset/bert-large-uncased-whole-word-masking-squad2"
  fi
  batch_size=${4:-"32"}
fi
if [ "$task_name" == "squad2" ] ; then
  dataset_name="squad_v2 --version_2_with_negative"
fi
mkdir -p ./log/bert_${size}_ptq_w3a4_g4_e2m6/$task_name
export CUDA_VISIBLE_DEVICES=$gpu_num
python run_qa.py \
  --do_eval \
  --model_name_or_path $path \
  --dataset_name $dataset_name \
  --max_seq_length 384 \
  --doc_stride 128 \
  --quantize_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --output_dir ./log/bert_${size}_ptq_w3a4_g4_e2m6/$task_name/ \
  --mode $mode \
  --abit 4 \
  --wbit 3 --w3_exp_bit 2 \
  -wu 250 \
  -wl 75  \
  -au 250 \
  -al 75 --group_size 4 2>&1 | tee ./log/bert_${size}_ptq_w3a4_g4_e2m6/$task_name/${batch_size}.log
