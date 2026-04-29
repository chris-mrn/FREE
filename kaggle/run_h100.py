import os
import subprocess

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# H100 supports the latest PyTorch — no need for the sm_60-pinned version
run("pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121")

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
# Upload fm_cifar10_weights_step_80000.pt as a Kaggle dataset and set this path
RESUME_CKPT = "/kaggle/input/<your-dataset>/fm_cifar10_weights_step_80000.pt"

# Resume training from step 80k to 400k
run(
    f"python train_cifar10.py"
    f" --model {MODEL}"
    f" --use_weight"
    f" --total_steps {TOTAL_STEPS + 1}"
    f" --save_step 20000"
    f" --batch_size 128"
    f" --num_workers 4"
    f" --output_dir {RESULTS_DIR}"
    f" --resume_ckpt {RESUME_CKPT}"
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
