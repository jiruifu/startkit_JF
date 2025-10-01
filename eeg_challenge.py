import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader
from braindecode.models.util import models_dict
from braindecode.models import *
import argparse
import sys
import os
from dependancy import trainer_challenge_1
from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset
from dependancy import setup_logger
from dependancy import challenge1, challenge2
from braindecode.models.util import models_dict
from dependancy import list_of_str, str2bool


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0)
        msg = f"CUDA-enabled GPU found. Training should be faster. GPU name: {gpu_name}"
    else:
        msg = "No CUDA-enabled GPU found. Training will be carried out on CPU."
    release_list = []
    for release in args.release:
        release_list.append(release)
    releasename = "-".join(release_list)
    exp_name = f"Challenge_{args.challenge}_{args.model}_{releasename}"
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(exp_name, "logs")
    logger.info(msg)
    names = sorted(models_dict)
    # for name, model in models_dict.items():
    #     print(f"Model: {name}")
    #     print(model)
    if args.model not in names:
        logger.error(f"Model {args.model} not found. Please choose from {names}")
        raise ValueError(f"Model {args.model} not found. Please choose from {names}")
    else:
        if args.challenge == "1":
            challenge1(epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    betas=args.betas,
                    device=device,
                    save_dir=args.save_dir,
                    logger=logger,
                    release=args.release,
                    full_data=args.full_data,
                    scheduler=args.scheduler,
                    early_stop=args.early_stop,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    model=args.model)
        elif args.challenge == "2":
            challenge2(epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    betas=args.betas,
                    device=device,
                    save_dir=args.save_dir,
                    logger=logger,
                    release=args.release,
                    full_data=args.full_data,
                    scheduler=args.scheduler,
                    early_stop=args.early_stop,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    model=args.model)

    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--challenge", type=str, default="1")
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--lr", type=float, default=0.0002)
    args.add_argument("--betas", type=tuple, default=(0.5, 0.9999))
    args.add_argument("--epochs", type=int, default=2000)
    args.add_argument("--model", type=str, default="EEGConformer")
    args.add_argument("--multi_gpu", type=str2bool, default=False)

    args.add_argument("--release", type=list_of_str, default=["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"])
    args.add_argument("--full_data", type=str2bool, default=True)
    args.add_argument("--scheduler", type=str2bool, default=False)
    args.add_argument("--early_stop", type=str2bool, default=True)
    args.add_argument("--patience", type=int, default=50)
    args.add_argument("--min_delta", type=float, default=1e-4)

    args.add_argument("--save_dir", type=str, default=r"/mnt/d/10_EEG-Challenge/startkit_JF/trained_models")
    args.add_argument("--log_dir", type=str, default=r"/mnt/d/10_EEG-Challenge/startkit_JF/logs")
    args.add_argument("--data_dir", type=str, default=r"/mnt/d/10_EEG-Challenge/startkit_JF/data")

    args = args.parse_args()
    main(args)