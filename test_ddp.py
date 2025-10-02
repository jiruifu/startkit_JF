"""
Simple PyTorch DistributedDataParallel (DDP) Training Script
This script trains pre-built CNN models (ResNet, VGG, MobileNet, EfficientNet) 
on CIFAR10 to test DDP functionality on a cluster.

Usage:
    Single node, multiple GPUs:
        torchrun --nproc_per_node=2 test_ddp.py --model resnet18 --epochs 10 --batch_size 128
    
    Multiple nodes (SLURM):
        See test_ddp.slurm for SLURM job script
        
Available models:
    resnet18, resnet34, resnet50, vgg11, vgg16, mobilenet_v2, efficientnet_b0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import argparse
import os
import time
from datetime import datetime


def setup_ddp():
    """Initialize the distributed environment."""
    # Check if we're running with torchrun or in SLURM environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun sets these automatically
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        
        # Setup master address and port for SLURM
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    else:
        # Single GPU or CPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_data_loaders(batch_size, rank, world_size):
    """Create train and test data loaders with DistributedSampler"""
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download dataset only on rank 0 to avoid conflicts
    if rank == 0:
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Wait for rank 0 to finish downloading
    if world_size > 1:
        dist.barrier()
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    
    # Create distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, device, rank, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0 and rank == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                  f'Loss: {running_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, test_loader, criterion, device, rank, world_size):
    """Validate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Aggregate metrics across all processes
    if world_size > 1:
        metrics = torch.tensor([test_loss, correct, total], dtype=torch.float32).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        test_loss, correct, total = metrics.cpu().numpy()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main(args):
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    if rank == 0:
        print(f"{'='*60}")
        print(f"PyTorch DistributedDataParallel Test")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"Device: {device}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Total Batch Size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning Rate: {args.lr}")
        print(f"{'='*60}\n")
    
    # Get data loaders
    train_loader, test_loader, train_sampler = get_data_loaders(args.batch_size, rank, world_size)
    
    # Create model using pre-built model from torchvision
    if args.model == 'resnet18':
        model = models.resnet18(weights=None)  # Use weights='DEFAULT' for pre-trained weights
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'vgg11':
        model = models.vgg11(weights=None)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
    elif args.model == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    if rank == 0:
        print(f"Model: {args.model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, rank, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device, rank, world_size)
        
        # Step scheduler
        scheduler.step()
        
        if rank == 0:
            print(f'\nEpoch {epoch}/{args.epochs}:')
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                if world_size > 1:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f'Best model saved with accuracy: {best_acc:.2f}%')
            print('-' * 60)
    
    # Training complete
    if rank == 0:
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.2f} minutes')
        print(f'Best Validation Accuracy: {best_acc:.2f}%')
    
    # Cleanup
    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDP Test Script')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg16', 
                                'mobilenet_v2', 'efficientnet_b0'],
                        help='pre-built model architecture to use')
    parser.add_argument('--save_dir', type=str, default='./ddp_checkpoints', 
                        help='directory to save checkpoints')
    
    args = parser.parse_args()
    main(args)

