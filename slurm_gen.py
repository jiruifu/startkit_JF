import os

def slurm_gen(model_name:str,
            release:list[str],
            time_limit:str,
            node_number:int,
            challenge:str,
            num_gpu:int,
            save_dir:str,
            data_dir:str,
            log_dir:str,
            lr:float,
            epochs:int,
            batch_size:int,
            multi_gpu:bool,
            scheduler:bool,
            early_stop:bool,
            full_data:bool,
            patience:int=50,
            min_delta:float=1e-4):
        job_name = f"{model_name}_{challenge}_{len(release)}"
        slurm_template = f"""#!/bin/bash

#SBATCH --job-name={job_name}   # Job name will be the experiment name
#SBATCH --output=%x_%j.out             # Name of output file (%x expands to job name, %j to jobId)
#SBATCH --error=%x_%j.err              # Name of error file (%x expands to job name, %j to jobId)
#SBATCH --nodes={node_number}                      # Run all processes on a single node
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --ntasks-per-node=1		
#SBATCH --time={time_limit}               # Time limit hrs:min:sec
#SBATCH --gres=gpu:{num_gpu}                  # Number of GPUs (per node)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ji045311@ucf.edu    # Where to send mail
echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"
# Load necessary modules (modify according to your system)
module purge
module load anaconda/anaconda-2023.09
conda activate eeg2025

# Navigate to the script directory
cd /lustre/fs1/home/ji045311/EEG_Challenge/
python eeg_challenge.py \\
    --challenge {challenge} \\
    --model {model_name} \\
    --release {','.join(release)} \\
    --full_data {full_data} \\
    --scheduler {scheduler} \\
    --early_stop {early_stop} \\
    --patience {patience} \\
    --min_delta {min_delta} \\
    --multi_gpu {multi_gpu} \\
    --save_dir {save_dir} \\
    --log_dir {log_dir} \\
    --data_dir {data_dir} \\
    --lr {lr} \\
    --epochs {epochs} \\
    --batch_size {batch_size}


"""
        return slurm_template

def main(path):
    model_name = "EEGConformer"
    release = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    time_limit = "24:00:00"
    node_number = 1
    challenge = ["1", "2"]
    num_gpu = 2
    save_dir = r"/lustre/fs1/home/ji045311/EEG_Challenge_Output/"
    data_dir = r"/lustre/fs1/home/ji045311/EEG_Challenge_Data/"
    log_dir = r"/lustre/fs1/home/ji045311/EEG_Challenge_log/"
    lr = 0.0002
    epochs = 1000
    batch_size = 256
    multi_gpu = True
    scheduler = False
    early_stop = True
    full_data = True
    patience = 50
    min_delta = 1e-4
    for challenge in challenge:
        slurm_template = slurm_gen(model_name=model_name, 
                                    release=release, 
                                    time_limit=time_limit, 
                                    node_number=node_number, 
                                    challenge=challenge, 
                                    num_gpu=num_gpu, 
                                    save_dir=save_dir, 
                                    data_dir=data_dir, 
                                    log_dir=log_dir, 
                                    lr=lr, 
                                    epochs=epochs, 
                                    batch_size=batch_size, 
                                    multi_gpu=multi_gpu, 
                                    scheduler=scheduler, 
                                    early_stop=early_stop, 
                                    full_data=full_data, 
                                    patience=patience, 
                                    min_delta=min_delta)
        slurm_fname = f"{model_name}_{challenge}_{len(release)}.slurm"
        slurm_path = os.path.join(path, slurm_fname)
        with open(slurm_path, 'w') as f:
            f.write(slurm_template)
        print(f"{model_name}_{challenge}_{len(release)}.slurm saved to {path}")

if __name__ == "__main__":
    path = r"/mnt/d/slurm"
    os.makedirs(path, exist_ok=True)
    main(path)