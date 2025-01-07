import argparse
import os
import random
import sys

import numpy as np
import torch
from easydict import EasyDict as edict

# Get paths and validate
try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(file_dir)
    workspace_root = os.path.join("/workspace")  # Docker mount point

    paths_to_add = [
        project_root,
        workspace_root,
        os.path.join(workspace_root, "mad")
    ]

    # Add paths if they exist and aren't already in sys.path
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to Python path: {path}")
        else:
            print(f"Path does not exist or already in sys.path: {path}")

    print(f"Project root: {project_root}")
    print(f"Python path: {os.environ.get('PYTHONPATH', '')}")

except Exception as e:
    print(f"Failed to setup paths: {str(e)}")
    raise

def get_config(args):

    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    config = edict(vars(args))

    if config.training_type == "MAD_training":
        config.use_lora = True
        config.train_scratch = False
    elif config.training_type == "MAD_training_only_header":
        config.use_lora = False
        config.train_scratch = False
    elif config.training_type == "MAD_training_scratch":
        config.use_lora = False
        config.train_scratch = True
    elif config.training_type == "test_clip":
        config.use_lora = False
        config.train_scratch = False

    dataset_paths = {
        "smdd": "/workspace/FaceMAD/Protocols/smdd_train_cropped.csv",
        "facemorpher": "/workspace/FaceMAD/Protocols/FaceMorpher.csv",
        "mipgan1": "/workspace/FaceMAD/Protocols/MIPGAN_I.csv",
        "mipgan2": "/workspace/FaceMAD/Protocols/MIPGAN_II.csv",
        "mordiff": "/workspace/FaceMAD/Protocols/MorDIFF.csv",
        "opencv": "/workspace/FaceMAD/Protocols/OpenCV.csv",
        "webmorph": "/workspace/FaceMAD/Protocols/Webmorph.csv",
        "feret_all_attacks": "/workspace/FaceMAD/evaluate_external/Protocols/feret_all_attacks.csv",
        "feret_morph_facemorpher": "/workspace/FaceMAD/evaluate_external/Protocols/feret_morph_facemorpher.csv",
        "feret_morph_opencv": "/workspace/FaceMAD/evaluate_external/Protocols/feret_morph_opencv.csv",
        "feret_morph_stylegan": "/workspace/FaceMAD/evaluate_external/Protocols/feret_morph_stylegan.csv",
        "frgc_all_attacks": "/workspace/FaceMAD/evaluate_external/Protocols/frgc_all_attacks.csv",
        "frgc_morph_facemorpher": "/workspace/FaceMAD/evaluate_external/Protocols/frgc_morph_facemorpher.csv",
        "frgc_morph_opencv": "/workspace/FaceMAD/evaluate_external/Protocols/frgc_morph_opencv.csv",
        "frgc_morph_stylegan": "/workspace/FaceMAD/evaluate_external/Protocols/frgc_morph_stylegan.csv",
        "frll_all_attacks": "/workspace/FaceMAD/evaluate_external/Protocols/frll_all_attacks.csv",
        "frll_morph_amsl": "/workspace/FaceMAD/evaluate_external/Protocols/frll_morph_amsl.csv",
        "frll_morph_facemorpher": "/workspace/FaceMAD/evaluate_external/Protocols/frll_morph_facemorpher.csv",
        "frll_morph_opencv": "/workspace/FaceMAD/evaluate_external/Protocols/frll_morph_opencv.csv",
        "frll_morph_stylegan": "/workspace/FaceMAD/evaluate_external/Protocols/frll_morph_stylegan.csv",
        "frll_morph_webmorph": "/workspace/FaceMAD/evaluate_external/Protocols/frll_morph_webmorph.csv"
    }

    if config.dataset_name == "MADTrain":
        config.dataset_path = dataset_paths["smdd"]
        # list of csvs for testing
        config.test_dataset_path = [dataset_paths["facemorpher"], dataset_paths["mipgan1"], dataset_paths["mipgan2"], dataset_paths["mordiff"], dataset_paths["opencv"], dataset_paths["webmorph"], dataset_paths["feret_all_attacks"], dataset_paths["feret_morph_facemorpher"], dataset_paths["feret_morph_opencv"], dataset_paths["feret_morph_stylegan"], dataset_paths["frgc_all_attacks"], dataset_paths["frgc_morph_facemorpher"], dataset_paths["frgc_morph_opencv"], dataset_paths["frgc_morph_stylegan"], dataset_paths["frll_all_attacks"], dataset_paths["frll_morph_amsl"], dataset_paths["frll_morph_facemorpher"], dataset_paths["frll_morph_opencv"], dataset_paths["frll_morph_stylegan"], dataset_paths["frll_morph_webmorph"]]

        config.test_data = ["facemorpher", "mipgan1", "mipgan2", "mordiff", "opencv", "webmorph", "feret_all_attacks", "feret_morph_facemorpher", "feret_morph_opencv", "feret_morph_stylegan", "frgc_all_attacks", "frgc_morph_facemorpher", "frgc_morph_opencv", "frgc_morph_stylegan", "frll_all_attacks", "frll_morph_amsl", "frll_morph_facemorpher", "frll_morph_opencv", "frll_morph_stylegan", "frll_morph_webmorph"]
    config.num_classes = 2
    if config.backbone_size == "ViT-B/16" or config.backbone_size == "ViT-B/32":
        # "ViT-B/32", "ViT-B/16", "ViT-L/14"
        config.training_desc = f'ViT-B16/{config.training_type}/{config.dataset_name}'
    elif config.backbone_size == "ViT-L/14":
        config.training_desc = f'ViT-L14/{config.training_type}/{config.dataset_name}'
    config.output_path = "/output/last_trainings/" + config.training_desc

    if config.training_type == "MAD_training":
        config.output_path = (
            f"{config.output_path}/lrm{config.lr_model:.0e}_lrh{config.lr_header:.0e}_d{config.lora_dropout:.0e}_a{config.lora_a}_r{config.lora_r}"
        )
    elif config.training_type == "MAD_training_only_header":
        config.output_path = f"{config.output_path}/lrh{config.lr_header:.0e}"
    elif config.training_type == "MAD_training_scratch":
        config.output_path = (
            f"{config.output_path}/lrm{config.lr_model:.0e}_lrh{config.lr_header:.0e}"
        )
    elif config.training_type == "test_clip":
        config.output_path = "/output/no_train/" + config.training_desc

    return config

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
    # cudnn.benchmark = True
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description="Distributed training job")
    parser.add_argument("--local-rank", type=int, help="local_rank")
    parser.add_argument(
        "--mode",
        default="training",
        choices=["training", "evaluation"],
        help="train or eval mode",
    )
    parser.add_argument(
        "--debug", default=False, type=bool, help="Log additional debug informations"
    )

    parser.add_argument("--backbone_size", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--training_type", type=str, required=True)

    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr_model", type=float, default=1e-6)
    parser.add_argument("--lr_header", type=float, default=1e-6)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=2)
    parser.add_argument("--lora_a", type=int, default=2)
    parser.add_argument("--max_norm", type=float, default=5)
    parser.add_argument("--loss", type=str, default="BinaryCrossEntropy")
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup", type=bool, default=True)
    parser.add_argument("--num_warmup_epochs", type=int, default=5)
    parser.add_argument("--T_0", type=int, default=5)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--lr_func_drop", type=int, nargs="+", default=[22, 30, 40])
    parser.add_argument("--batch_size", type=int, default=86)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q", "v"]
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--normalize_type", type=str, default="clip")
    parser.add_argument("--interpolation_type", type=str, default="bicubic")
    parser.add_argument(
        "--eval_path", type=str, default="/home/chettaou/data/validation"
    )
    parser.add_argument("--val_targets", type=str, nargs="+", default=[])
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=40)
    parser.add_argument("--batch_size_eval", type=int, default=16)
    parser.add_argument("--horizontal_flip", type=bool, default=True)
    parser.add_argument("--rand_augment", type=bool, default=True)
    args = parser.parse_args()
    config = get_config(args)
    from src.train import main
    main(config)
