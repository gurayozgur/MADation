#!/usr/bin/bash

export OMP_NUM_THREADS=2
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_HOME="./cache/"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

declare -a backbone_sizes=("ViT-B/16" "ViT-L/14")
declare -a training_types=("test_clip")

for backbone_size in "${backbone_sizes[@]}"; do
  for training_type in "${training_types[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
      --nproc_per_node=2 \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr="localhost" \
      --master_port=12355 \
      src/config.py \
      --debug=True \
      --backbone_size="$backbone_size" \
      --dataset_name="MADTrain" \
      --training_type="$training_type"
  done
done
