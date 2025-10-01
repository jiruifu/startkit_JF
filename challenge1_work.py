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
from trainer import trainer_challenge_1
from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        msg ='CUDA-enabled GPU found. Training should be faster.'
    else:
        msg = (
            "No GPU found. Training will be carried out on CPU, which might be "
            "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
            " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
            "selecting \'T4 GPU\'\nunder \'Hardware accelerator\'."
        )
    print(msg)

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)


    dataset_ccd = EEGChallengeDataset(task="contrastChangeDetection",
                                    release="R5", cache_dir=DATA_DIR,
                                    mini=False)
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
    sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
            "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
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
    
    batch_size = args.batch_size
    num_workers = 1 # We are using a single worker, but you can increase this for faster data loading

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = EEGConformer(n_chans=129, n_outputs=1, n_times=200, sfreq=100)
    print(model)
    model.to(device)
    lr = args.lr
    betas = args.betas
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
    trained_model = trainer_challenge_1(train_loader=train_loader,
                        valid_loader=valid_loader, 
                        model=model, 
                        loss_fn=loss_fn, 
                        optimizer=optimizer, 
                        device=device, 
                        scheduler=False, 
                        early_stop=False, 
                        n_epochs=args.epochs,
                        print_batch_stats=True)
    model_name = f"{model.__class__.__name__}_challenge_1.pt"
    torch.save(trained_model.state_dict(), model_name)
    print(f"Model saved as {model_name}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--betas", type=tuple, default=(0.5, 0.9999))
    parser.add_argument("--epochs", type=int, default=500)

    args = parser.parse_args()
    main(args)