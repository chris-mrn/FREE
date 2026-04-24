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

# 10-step smoke test: weighting precomputed, images saved at steps 0 and 5
run(
    "python train_cifar10.py"
    " --model fm"
    " --use_weight"
    " --total_steps 10"
    " --save_step 5"
    " --batch_size 128"
    " --num_workers 2"
    " --output_dir /kaggle/working/results/"
)

print("Test complete. Check /kaggle/working/results/fm/ for weights and images.")
