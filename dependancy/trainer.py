import copy
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
import logging
import traceback
import torch.nn as nn
def trainer_challenge_1(train_loader: DataLoader,
                            valid_loader: DataLoader, 
                            model: Module, 
                            loss_fn, 
                            optimizer, 
                            device,
                            scheduler:bool = False,
                            early_stop:bool = False,
                            n_epochs = 100,
                            patience = 5,
                            min_delta = 1e-4,
                            logger: logging.Logger = None,
                            print_freq: int = 5,
                            print_instant:bool = False,
                            miniBatch_norm:bool = True,
                            batch_time_norm:bool = False,
                            multi_gpu:bool = False)->Module:
    def train_one_epoch():
        model.train()
        total_loss = 0.0
        sum_sq_err = 0.0
        n_samples = 0
        for i, train_mini_batch in enumerate(train_loader):
            # Support datasets that may return (X, y) or (X, y, ...)
            X, y = train_mini_batch[0].to(device).float(), train_mini_batch[1].to(device).float()

            if miniBatch_norm:
                for i in range(X.shape[0]):
                    mean = X[i,:,:].mean()
                    std = X[i,:,:].std()
                    X[i,:,:] = (X[i,:,:] - mean) / (std + 1e-8)
            elif batch_time_norm:
                mean = X.mean(dim=-1, keepdim=True)
                std = X.std(dim=-1, keepdim=True)
                X = (X - mean) / (std + 1e-8)
            else:
                raise ValueError("Invalid normalization type")
            
            optimizer.zero_grad(set_to_none=True)
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Flatten to 1D for regression metrics and accumulate squared error
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if i % print_freq == 0 and print_instant:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                logger.info(f"Epoch {epoch}/{n_epochs}, Batch {i + 1}/{len(train_loader)}, "
                    f"Instant Train Loss: {loss.item():.6f}, Instant Train RMSE: {running_rmse:.6f}")

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
        return avg_loss, rmse
    
    @torch.no_grad()
    def valid_model():
        model.eval()

        total_loss = 0.0
        sum_sq_err = 0.0
        n_batches = len(valid_loader)
        n_samples = 0

        for i, valid_mini_batch in enumerate(valid_loader):
            # Supports (X, y) or (X, y, ...)
            X, y = valid_mini_batch[0].to(device).float(), valid_mini_batch[1].to(device).float()
            
            preds = model(X)
            batch_loss = loss_fn(preds, y).item()
            total_loss += batch_loss

            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if i % print_freq == 0 and print_instant:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                logger.info(f"Epoch {epoch}/{n_epochs}, Batch {i + 1}/{len(valid_loader)}, "
                    f"Instant Val Loss: {batch_loss:.6f}, Instant Val RMSE: {running_rmse:.6f}")


        avg_loss = total_loss / n_batches if n_batches else float("nan")
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
        return avg_loss, rmse
    try:
        if multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        optimizer = optimizer
        if scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
        scheduler = None
        loss_fn = loss_fn

        best_rmse = float("inf")
        epochs_no_improve = 0
        best_state, best_epoch = None, None
        logger.info(f"Training challenge 1 model for {n_epochs} epochs")
        logger.info(f"Using the model: {model.__class__.__name__}")
        logger.info(f"Using the loss function: {loss_fn.__class__.__name__}")
        logger.info(f"Using the optimizer: {optimizer.__class__.__name__}")
        if scheduler:
            logger.info(f"Using the scheduler: {scheduler.__class__.__name__}")
        else:
            logger.info("Using no scheduler")
        logger.info(f"Using the device: {device}")
        if early_stop:
            logger.info(f"Using early stopping with patience: {patience} and min_delta: {min_delta}")
        else:
            logger.info("Using no early stopping")

        for epoch in range(1, n_epochs + 1):

            train_loss, train_rmse = train_one_epoch()
            val_loss, val_rmse = valid_model()

            logger.info(f"Epoch {epoch}/{n_epochs}, Train RMSE: {train_rmse:.6f}, Train Loss: {train_loss:.6f}, Val RMSE: {val_rmse:.6f}, Val Loss: {val_loss:.6f}\n")
            if early_stop:
                if val_rmse < best_rmse - min_delta:
                    best_rmse = val_rmse
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info(f"Early stopping at epoch {epoch}/{n_epochs}. Best Val RMSE: {best_rmse:.6f} at (epoch {best_epoch})")
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.error(traceback.format_exc())
        raise e


def trainer_challenge_2(dataloader: DataLoader,
                        model: Module,
                        optimizer,
                        loss_fn,
                        device,
                        n_epochs = 2,
                        logger: logging.Logger = None,
                        print_freq: int = 5,
                        multi_gpu:bool = False)->Module:
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(1, n_epochs + 1):
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            X, y, crop_inds, infos = batch
            X = X.to(device).float()
            y = y.to(device).float().unsqueeze(1)
            preds = model(X)
            loss = loss_fn(preds, y)
            if idx % print_freq == 0:
                logger.info(f"Epoch {epoch}/{n_epochs}, Batch {idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}")
            loss.backward()
            optimizer.step()
    return model
