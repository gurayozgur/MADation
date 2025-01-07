export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_HOME="./cache/"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=12355 \
  mad/src/config.py --debug=True \
  --backbone_size="ViT-B/16" \
  --dataset_name="MADTrain" \
  --training_type="MAD_training" \
  --lr_model=1e-5 \
  --lr_header=1e-4 \
  --lora_dropout=0.4 \
  --lora_r=2 \
  --lora_a=4 \
  --batch_size=32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=12355 \
  mad/src/config.py --debug=True \
  --backbone_size="ViT-L/14" \
  --dataset_name="MADTrain" \
  --training_type="MAD_training" \
  --lr_model=1e-5 \
  --lr_header=1e-4 \
  --lora_dropout=0.2 \
  --lora_r=2 \
  --lora_a=8 \
  --batch_size=32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=12355 \
  mad/src/config.py --debug=True \
  --backbone_size="ViT-B/16" \
  --dataset_name="MADTrain" \
  --training_type="MAD_training_only_header" \
  --lr_model=0 \
  --lr_header=1e-2 \
  --batch_size=32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=12355 \
  mad/src/config.py --debug=True \
  --backbone_size="ViT-L/14" \
  --dataset_name="MADTrain" \
  --training_type="MAD_training_only_header" \
  --lr_model=0 \
  --lr_header=1e-2 \
  --batch_size=32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=12355 \
  mad/src/config.py --debug=True \
  --backbone_size="ViT-B/16" \
  --dataset_name="MADTrain" \
  --training_type="MAD_training_scratch" \
  --lr_model=1e-5 \
  --lr_header=5e-5 \
  --batch_size=32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=1 \
  --node_rank=0 --master_addr="localhost" --master_port=12355 \
  mad/src/config.py --debug=True \
  --backbone_size="ViT-L/14" \
  --dataset_name="MADTrain" \
  --training_type="MAD_training_scratch" \
  --lr_model=1e-4 \
  --lr_header=1e-4 \
  --batch_size=32


