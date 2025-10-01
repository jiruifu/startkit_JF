from pathlib import Path
import argparse
import logging
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch import optim
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from typing import Tuple, List

from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.base import EEGWindowsDataset, BaseDataset
from braindecode.models import *
from braindecode.models.util import models_dict
from braindecode.preprocessing import (
    create_fixed_length_windows,
    preprocess,
    Preprocessor,
    create_windows_from_events
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

from .trainer import trainer_challenge_1, trainer_challenge_2


global sub_rm
sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
          "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]

class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        # P-factor label:
        p_factor = self.dataset.description["p_factor"]
        p_factor = float(p_factor)

        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, p_factor, (i_window_in_trial, i_start, i_stop), infos

def challenge1(epochs:int, 
                batch_size:int, 
                lr:float, 
                betas:Tuple[float, float], 
                device:str,
                save_dir:str,
                data_dir:str,
                logger:logging.Logger,
                release:List[str]=["R5", "R6"],
                full_data:bool=True,
                scheduler:bool=False,
                early_stop:bool=False,
                patience:int=5,
                min_delta:float=1e-4,
                model:str="EEGConformer"
                ):

    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    release_list = release
    all_datasets_list = [
        EEGChallengeDataset(task="contrastChangeDetection",
                            release=release, cache_dir=DATA_DIR,
                            mini=not full_data)
        for release in release_list
    ]
    dataset_ccd = BaseConcatDataset(all_datasets_list)
    logger.info(dataset_ccd.description)
    raws = Parallel(n_jobs=-1)(
        delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets
    )
    EPOCH_LEN_S = 2.0
    SFREQ = 100 # by definition here

    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
            require_stimulus=True, require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_ccd, transformation_offline, n_jobs=1)

    ANCHOR = "stimulus_anchor"

    SHIFT_AFTER_STIM = 0.5
    WINDOW_LEN       = 2.0

    # Keep only recordings that actually contain stimulus anchors
    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

    # Create single-interval windows (stim-locked, long enough to include the response)
    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),                 # +0.5 s
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),   # +2.5 s
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    # Injecting metadata into the extra mne annotation.
    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
            "stimulus_onset", "response_onset", "correct", "response_type")
            )
    meta_information = single_windows.get_metadata()
    valid_frac = 0.1
    test_frac = 0.1
    seed = 2025

    subjects = meta_information["subject"].unique()
    subjects = [s for s in subjects if s not in sub_rm]

    train_subj, valid_test_subject = train_test_split(
        subjects, test_size=(valid_frac + test_frac), random_state=check_random_state(seed), shuffle=True
    )

    valid_subj, test_subj = train_test_split(
        valid_test_subject, test_size=test_frac, random_state=check_random_state(seed + 1), shuffle=True
    )
    # sanity check
    assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)
    subject_split = single_windows.split("subject")

    train_set = []
    valid_set = []
    test_set = []

    for s in subject_split:
        if s in train_subj:
            train_set.append(subject_split[s])
        elif s in valid_subj:
            valid_set.append(subject_split[s])
        elif s in test_subj:
            test_set.append(subject_split[s])

    train_set = BaseConcatDataset(train_set)
    valid_set = BaseConcatDataset(valid_set)
    test_set = BaseConcatDataset(test_set)

    print("Number of examples in each split in the minirelease")
    print(f"Train:\t{len(train_set)}")
    print(f"Valid:\t{len(valid_set)}")
    print(f"Test:\t{len(test_set)}")
    
    batch_size = batch_size
    num_workers = 1 # We are using a single worker, but you can increase this for faster data loading

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = models_dict[model](n_chans=129, n_outputs=1, n_times=200, sfreq=100)
    print(model)
    model.to(device)
    lr = lr
    betas = betas
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
    trained_model = trainer_challenge_1(train_loader=train_loader,
                        valid_loader=valid_loader, 
                        model=model, 
                        loss_fn=loss_fn, 
                        optimizer=optimizer, 
                        device=device, 
                        logger=logger,
                        scheduler=scheduler, 
                        early_stop=early_stop, 
                        n_epochs=epochs,
                        print_freq=5,
                        patience=patience,
                        min_delta=min_delta)
    model_name = f"{model.__class__.__name__}_challenge_1.pt"
    model_name = os.path.join(save_dir, model_name)
    torch.save(trained_model.state_dict(), model_name)
    print(f"Model saved as {model_name}")


def challenge2(epochs:int, 
                batch_size:int, 
                lr:float, 
                betas:Tuple[float, float], 
                device:str,
                save_dir:str,
                data_dir:str,
                logger:logging.Logger,
                release:List[str]=["R5", "R6"],
                full_data:bool=True,
                scheduler:bool=False,
                early_stop:bool=False,
                patience:int=5,
                min_delta:float=1e-4,
                model:str="EEGConformer"
                ):
    # The first step is define the cache folder!
    DATA_DIR = Path(data_dir)

    # Creating the path if it does not exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # We define the list of releases to load.
    # Here, only release 5 is loaded.
    release_list = release

    all_datasets_list = [
        EEGChallengeDataset(
            release=release,
            task="contrastChangeDetection",
            mini=False,
            description_fields=[
                "subject",
                "session",
                "run",
                "task",
                "age",
                "gender",
                "sex",
                "p_factor",
            ],
            cache_dir=DATA_DIR,
        )
        for release in release_list
    ]
    logger.info("Datasets loaded")
    all_datasets = BaseConcatDataset(all_datasets_list)
    logger.info(all_datasets.description)

    raws = Parallel(n_jobs=os.cpu_count())(
        delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
    )

    SFREQ = 100
    # Filter out recordings that are too short
    all_datasets = BaseConcatDataset(
        [
            ds
            for ds in all_datasets.datasets
            if not ds.description.subject in sub_rm
            and ds.raw.n_times >= 4 * SFREQ
            and len(ds.raw.ch_names) == 129
            and not math.isnan(ds.description["p_factor"])
        ]
    )

    # Create 4-seconds windows with 2-seconds stride
    windows_ds = create_fixed_length_windows(
        all_datasets,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )

    # Wrap each sub-dataset in the windows_ds
    windows_ds = BaseConcatDataset(
        [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
    ) 
    model = models_dict[model](n_chans=129, n_outputs=1, n_times=2 * SFREQ).to(device)
    optimizer = torch.optim.Adamax(params=model.parameters(), lr=lr, betas=betas)
    logger.info(model)
    loss_fn = l1_loss()

    num_workers = 1 # We are using a single worker, but you can increase this for faster data loading
    dataloader = DataLoader(windows_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    trained_model = trainer_challenge_2(dataloader=dataloader,
                                        model=model,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        device=device,
                                        n_epochs=epochs,
                                        logger=logger,
                                        print_freq=5)