import os
import subprocess

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Install PyTorch compatible with P100 (sm_60, dropped in PyTorch 2.4+)
run("pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121")

# Install dependencies
run("pip install -q absl-py torchdyn pot clean-fid torchdiffeq")

# Clone repo
run("git clone https://github.com/chris-mrn/FREE.git")

# Install torchcfm from source
run("pip install -q -e FREE/")

os.chdir("FREE/examples/images/cifar10")

RESULTS_DIR = "/kaggle/working/results/"
MODEL = "fm"
TOTAL_STEPS = 400000

# Train
# Architecture: UNet (not Diffusion Transformer)
# Weights saved every 20000 steps under RESULTS_DIR/fm/
# Generated image grids saved alongside each checkpoint
run(
    f"python train_cifar10.py"
    f" --model {MODEL}"
    f" --use_weight"
    f" --total_steps {TOTAL_STEPS + 1}"
    f" --save_step 20000"
    f" --batch_size 128"
    f" --num_workers 2"
    f" --output_dir {RESULTS_DIR}"
)

# Compute FID with the final EMA checkpoint
run(
    f"python compute_fid.py"
    f" --model {MODEL}"
    f" --step {TOTAL_STEPS}"
    f" --input_dir {RESULTS_DIR}"
    f" --integration_method dopri5"
    f" --num_gen 50000"
    f" --batch_size_fid 1024"
)
