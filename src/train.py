import argparse
import os
import random
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.getcwd()))

from backbone import get_model, get_output_dim
from data.loaders import DataLoaderX
from data.transform import transform_image
from finetuning import apply_lora_model
from training import get_header, get_trainer
from utils.logging import TrainingLogger

from data import get_dataset

torch.backends.cudnn.benchmark = True

os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

def main(cfg):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) # args.local_rank
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    # Logging
    TrainingLogger(local_rank, cfg.output_path)

    # Dataset
    trainset, testset = get_dataset(local_rank, **cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    dataloader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        pin_memory=True, drop_last=True, num_workers=0, sampler=train_sampler
    )
    # testset a list of datasets, create a sampler for each, and a dataloader for each
    test_sampler = []
    test_dataloader = []
    for test in testset:
        test_sampler.append(torch.utils.data.distributed.DistributedSampler(test, shuffle=False, drop_last=False))
        test_dataloader.append(DataLoaderX(
            local_rank=local_rank, dataset=test, batch_size=cfg.batch_size,
            pin_memory=True, drop_last=False, num_workers=0, sampler=test_sampler[-1]
        ))

    # Model
    model = get_model(local_rank, **cfg)
    if cfg.use_lora: # LoRA
        apply_lora_model(local_rank, model, **cfg)
    elif cfg.train_scratch:
        model.backbone.initialize_parameters()
        print("Model initialized from scratch")
    model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    if cfg.use_lora or cfg.train_scratch:
        model.train()

    # Header
    output_dim = get_output_dim(**cfg)
    header = get_header(rank=local_rank, backbone_out_dim=output_dim, **cfg).to(local_rank)
    header = DistributedDataParallel(module=header, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    header.train()

    if cfg.training_type == "MAD_training" or cfg.training_type == "MAD_training_scratch":
        traintype = "MAD_training"
    elif cfg.training_type == "MAD_training_only_header":
        traintype = "MAD_training_only_header"
    elif cfg.training_type == "test_clip":
        traintype = "test_clip"
    else:
        ValueError()

    # Training
    model_trainer = get_trainer(
        rank=local_rank,
        world_size=world_size,
        model_name=cfg.model_name,
        model=model,
        trainset=trainset,
        dataloader=dataloader,
        train_sampler=train_sampler,
        training_type=traintype,
        config=cfg,
        header=header,
        test_dataloader=test_dataloader,
        test_sampler=test_sampler
    )
    model_trainer.start_training()

    if local_rank == 0:
        destroy_process_group()

'''
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":

    torch.cuda.empty_cache()
    #cudnn.benchmark = True
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--local-rank', type=int, help='local_rank')
    parser.add_argument('--mode', default='training', choices=['training', 'evaluation'], help='train or eval mode')
    parser.add_argument('--debug', default=False, type=bool, help='Log additional debug informations')
    args = parser.parse_args()

    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    main(args)
'''
