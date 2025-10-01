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
                            epochs_no_improve = 0,
                            print_batch_stats: bool = True)->Module:
    def train_one_epoch(dataloader: DataLoader,
                         model: Module,
                         loss_fn,
                         optimizer,
                         scheduler: Optional[LRScheduler],
                         epoch: int,
                         device,
                         print_batch_stats: bool = True):
        model.train()
        total_loss = 0.0
        sum_sq_err = 0.0
        n_samples = 0
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not print_batch_stats
        )
        for batch_idx, batch in progress_bar:
            # Support datasets that may return (X, y) or (X, y, ...)
            X, y = batch[0], batch[1]
            X, y = X.to(device).float(), y.to(device).float()
            
            # Z-score normalization: per-channel standardization
            mean = X.mean(dim=2, keepdim=True)  # mean across time dimension
            std = X.std(dim=2, keepdim=True)    # std across time dimension
            X = (X - mean) / (std + 1e-8)       # standardize (epsilon avoids division by zero)

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

            if print_batch_stats:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                progress_bar.set_description(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
                )

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
        return avg_loss, rmse
    
    @torch.no_grad()
    def valid_model(
        dataloader: DataLoader,
        model: Module,
        loss_fn,
        device,
        print_batch_stats: bool = True,
    ):
        model.eval()

        total_loss = 0.0
        sum_sq_err = 0.0
        n_batches = len(dataloader)
        n_samples = 0

        iterator = tqdm(
            enumerate(dataloader),
            total=n_batches,
            disable=not print_batch_stats
        )

        for batch_idx, batch in iterator:
            # Supports (X, y) or (X, y, ...)
            X, y = batch[0], batch[1]
            X, y = X.to(device).float(), y.to(device).float()
            
            # Z-score normalization: per-channel standardization
            mean = X.mean(dim=2, keepdim=True)  # mean across time dimension
            std = X.std(dim=2, keepdim=True)    # std across time dimension
            X = (X - mean) / (std + 1e-8)       # standardize (epsilon avoids division by zero)

            preds = model(X)
            batch_loss = loss_fn(preds, y).item()
            total_loss += batch_loss

            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if print_batch_stats:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                iterator.set_description(
                    f"Val Batch {batch_idx + 1}/{n_batches}, "
                    f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
                )

        avg_loss = total_loss / n_batches if n_batches else float("nan")
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

        print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
        return avg_loss, rmse
    
    optimizer = optimizer
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
    scheduler = None
    loss_fn = loss_fn

    best_rmse = float("inf")
    epochs_no_improve = epochs_no_improve
    best_state, best_epoch = None, None

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}: ", end="")

        train_loss, train_rmse = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device
        )
        val_loss, val_rmse = valid_model(valid_loader, model, loss_fn, device)

        print(
            f"Train RMSE: {train_rmse:.6f}, "
            f"Average Train Loss: {train_loss:.6f}, "
            f"Val RMSE: {val_rmse:.6f}, "
            f"Average Val Loss: {val_loss:.6f}"
        )
        if early_stop:
            if val_rmse < best_rmse - min_delta:
                best_rmse = val_rmse
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
    return model