#!/bin/bash
task_name=${1:-"cola"}
size=${2:-"base"}
gpu_num=${3:-0}
batch_size=${4:-"128"}
mode=${5:-"ant-int-flint"}
if [ "$size" == "base" ] ; then
  path="ModelTC/bart-base-$task_name "
fi
mkdir -p ./log/bart_${size}_ptq_w2a4_g8/$task_name
export CUDA_VISIBLE_DEVICES=$gpu_num
python run_glue.py \
  --do_eval \
  --model_name_or_path $path \
  --task_name $task_name \
  --max_length 128 \
  --quantize_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --output_dir ./log/bart_${size}_ptq_w2a4_g8/$task_name/ \
  --mode $mode \
  --abit 4 \
  --wbit 2 \
  -wu 250 \
  -wl 75  \
  -au 250 \
  -al 75 --group_size 8 2>&1 | tee ./log/bart_${size}_ptq_w2a4_g8/$task_name/${batch_size}.log
