# PyTorch DistributedDataParallel (DDP) Test

This is a simple test script to verify that DistributedDataParallel works correctly on your cluster setup.

## Files

- `test_ddp.py` - Main training script with DDP support
- `test_ddp.slurm` - SLURM job script for cluster execution
- `DDP_TEST_README.md` - This file

## What the Script Does

- Trains pre-built CNN models (ResNet, VGG, MobileNet, EfficientNet) on CIFAR10 dataset
- Uses PyTorch's DistributedDataParallel (DDP) for multi-GPU training
- Saves the best model checkpoint
- Reports training and validation metrics

## Usage

### Option 1: Local Testing (Single Machine, Multiple GPUs)

```bash
# Using 2 GPUs on a single machine with ResNet18
torchrun --nproc_per_node=2 test_ddp.py --model resnet18 --epochs 10 --batch_size 128

# Using 4 GPUs with ResNet50
torchrun --nproc_per_node=4 test_ddp.py --model resnet50 --epochs 10 --batch_size 128

# Try different models
torchrun --nproc_per_node=2 test_ddp.py --model mobilenet_v2 --epochs 10 --batch_size 128
```

### Option 2: Single GPU or CPU (No DDP)

```bash
python test_ddp.py --model resnet18 --epochs 10 --batch_size 128
```

### Option 3: Cluster with SLURM

1. **Edit the SLURM script** (`test_ddp.slurm`):
   - Adjust the paths to match your cluster setup
   - Modify `--gres=gpu:X` to request the number of GPUs you want
   - Modify `--ntasks-per-node` to match the number of GPUs
   - Update `--mail-user` with your email
   - Change the working directory path if needed

2. **Submit the job**:
   ```bash
   sbatch test_ddp.slurm
   ```

3. **Check job status**:
   ```bash
   squeue -u $USER
   ```

4. **View output**:
   ```bash
   # While running
   tail -f test_ddp_<JOBID>.out
   
   # After completion
   cat test_ddp_<JOBID>.out
   cat test_ddp_<JOBID>.err
   ```

## Script Arguments

- `--model`: Pre-built model architecture to use (default: resnet18)
  - Available options: `resnet18`, `resnet34`, `resnet50`, `vgg11`, `vgg16`, `mobilenet_v2`, `efficientnet_b0`
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size per GPU (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save model checkpoints (default: ./ddp_checkpoints)

## Expected Behavior

When DDP is working correctly, you should see:

1. **Multiple processes initialized**: World size > 1
2. **Distributed sampler active**: Each GPU gets different data
3. **Synchronized validation metrics**: All GPUs report the same validation accuracy
4. **Linear speedup**: Training time should decrease roughly proportionally with number of GPUs

## Example Output

```
============================================================
PyTorch DistributedDataParallel Test
============================================================
World Size: 2
Device: cuda:0
Batch Size per GPU: 128
Total Batch Size: 256
Epochs: 20
Learning Rate: 0.001
============================================================

Model: resnet18
Total parameters: 11,173,962
Trainable parameters: 11,173,962

Epoch: 1 | Batch: 0/196 | Loss: 2.304 | Acc: 8.59%
Epoch: 1 | Batch: 50/196 | Loss: 2.156 | Acc: 16.24%
...

Epoch 1/20:
Train Loss: 1.752 | Train Acc: 35.45%
Val Loss: 1.523 | Val Acc: 44.32%
Best model saved with accuracy: 44.32%
------------------------------------------------------------
```

## Troubleshooting

### Issue: NCCL initialization fails
- **Solution**: Make sure CUDA and NCCL are properly installed
- Check that GPUs are visible: `nvidia-smi`

### Issue: "Address already in use"
- **Solution**: Change `MASTER_PORT` in the SLURM script or environment

### Issue: Different validation accuracies across ranks
- **Solution**: This indicates DDP synchronization is not working correctly
- Check that `dist.all_reduce` is being called during validation

### Issue: Data downloading conflicts
- **Solution**: The script handles this by having rank 0 download first, then using `dist.barrier()`

## Available Pre-built Models

The script supports these pre-built PyTorch models:
- **ResNet**: `resnet18`, `resnet34`, `resnet50` (good for general-purpose testing)
- **VGG**: `vgg11`, `vgg16` (deeper but slower)
- **MobileNet**: `mobilenet_v2` (lightweight, good for limited resources)
- **EfficientNet**: `efficientnet_b0` (modern, efficient architecture)

## Modifying for Your Own Models

To adapt this script for your own models:

1. Add your model architecture to the model selection section (around line 207-230)
2. Replace CIFAR10 dataset with your data in `get_data_loaders()`
3. Adjust the loss function and optimizer as needed
4. Keep the DDP setup and cleanup code the same (setup_ddp, cleanup_ddp, DDP wrapper)

## Performance Notes

- **Batch size**: The specified batch size is per GPU. Total effective batch size = batch_size × num_gpus
- **Number of workers**: Set to 4 per GPU, adjust based on your CPU count
- **Pin memory**: Enabled for faster data transfer to GPU
- **Gradient synchronization**: Happens automatically in DDP during `loss.backward()`

## Next Steps

Once this test works successfully, you can integrate DDP into your main `eeg_challenge.py` script using the same patterns.

