import subprocess
import sys
import os

def run(cmd, **kwargs):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)

# Install dependencies
run("pip install -q absl-py torchdyn pot clean-fid")

# Clone the repo
run("git clone https://github.com/chris-mrn/FREE.git")

# Install torchcfm from the cloned repo
run("pip install -q -e FREE/")

# Move into the CIFAR10 training directory
os.chdir("FREE/examples/images/cifar10")

# Run training with time-dependent weighting
run(
    "python train_cifar10.py"
    " --model fm"
    " --use_weight"
    " --total_steps 400001"
    " --batch_size 128"
    " --num_workers 2"
    " --output_dir /kaggle/working/results/"
)
