"""
Detect memorization in generated images by comparing to training images.

Computes embeddings (Inception) for all images and analyzes nearest-neighbor
distances to flag potential memorization.

Usage:
  python scripts/detect_memorization.py \\
      --train_dir path/to/train/images \\
      --gen_dir path/to/generated/images \\
      --out_dir outputs/memorization

Outputs:
  - memorization_report.txt: summary of findings
  - distance_histogram.png: histogram of nearest-neighbor distances
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Inception for embeddings
from torchvision.models import inception_v3
from torchvision import transforms

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')


def extract_images_from_grid(grid_path, grid_size=8):
    """Extract individual images from a grid PNG."""
    grid_img = Image.open(grid_path).convert('RGB')
    w, h = grid_img.size
    cell_w = w // grid_size
    cell_h = h // grid_size

    images = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_w
            y = row * cell_h
            cell = grid_img.crop((x, y, x + cell_w, y + cell_h))
            images.append(cell)

    return images


def load_images(image_dir, max_images=None, device='cpu'):
    """Load images from directory and return as tensors."""
    from torchvision import datasets

    image_dir = Path(image_dir)

    # If it's a special keyword, load from dataset
    if str(image_dir) == 'cifar10':
        print("Loading CIFAR-10 training images directly from PyTorch...")
        ds = datasets.CIFAR10(
            root='/tmp/cifar10_dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        )
        if max_images:
            ds = torch.utils.data.Subset(ds, range(min(max_images, len(ds))))
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
        images = []
        for batch, _ in tqdm(loader, desc="Loading CIFAR-10 images"):
            images.append(batch)
        images = torch.cat(images, 0).to(device)
        print(f"Loaded {len(images)} CIFAR-10 training images")
        return images

    # Otherwise, try to load from directory
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    if not image_files:
        raise ValueError(f"No images found in {image_dir}\n"
                        f"Tip: Use --train_dir cifar10 to load CIFAR-10 training set directly")

    print(f"Found {len(image_files)} files in {image_dir}")

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    images = []

    # Check if these are grid images (e.g., samples_step*.png with 8x8 grids)
    is_grid = image_files[0].name.startswith('samples_step') and image_files[0].suffix == '.png'

    if is_grid:
        print(f"Detected grid format. Extracting individual images from grids...")
        for img_path in tqdm(image_files, desc="Extracting from grids"):
            try:
                cell_images = extract_images_from_grid(img_path, grid_size=8)
                for cell_img in cell_images:
                    img_tensor = transform(cell_img)
                    images.append(img_tensor)
                    if max_images and len(images) >= max_images:
                        break
            except Exception as e:
                print(f"  Warning: Failed to process {img_path}: {e}")
            if max_images and len(images) >= max_images:
                break
    else:
        for img_path in tqdm(image_files, desc="Loading images"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                if max_images and len(images) >= max_images:
                    break
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")

    if not images:
        raise ValueError(f"No images loaded from {image_dir}")

    images = torch.stack(images).to(device)
    print(f"Loaded {len(images)} total images")
    return images


def get_inception_embeddings(images, device='cpu', batch_size=64):
    """
    Extract embeddings from Inception V3 (pre-logits layer).

    Args:
        images: (N, 3, 299, 299) tensor
        device: device
        batch_size: batch size for processing

    Returns:
        embeddings: (N, 2048) numpy array
    """
    # Load pretrained Inception V3
    from torchvision.models import Inception_V3_Weights
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    inception.eval()

    # Hook to extract pre-logits
    embeddings_list = []

    def hook_fn(module, input, output):
        embeddings_list.append(output.detach().cpu())

    # Register hook on last avgpool layer
    inception.avgpool.register_forward_hook(hook_fn)

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Computing embeddings"):
            batch = images[i:i+batch_size].to(device)
            _ = inception(batch)

    embeddings = torch.cat(embeddings_list, dim=0).numpy()  # (N, 2048)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Flatten if needed

    return embeddings


def compute_nearest_neighbors(gen_embeddings, train_embeddings):
    """
    For each generated image, find nearest training image.

    Args:
        gen_embeddings: (n_gen, D) array
        train_embeddings: (n_train, D) array

    Returns:
        distances: (n_gen,) array of nearest neighbor distances
        indices: (n_gen,) array of nearest neighbor indices
    """
    print("Computing nearest neighbors...")

    # Normalize embeddings
    gen_norm = gen_embeddings / (np.linalg.norm(gen_embeddings, axis=1, keepdims=True) + 1e-8)
    train_norm = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)

    # Cosine distance: 1 - cosine_similarity
    # Using batch processing for memory efficiency
    distances = []
    indices = []

    batch_size = 256
    for i in tqdm(range(0, len(gen_norm), batch_size), desc="Finding nearest neighbors"):
        batch = gen_norm[i:i+batch_size]
        # Cosine similarity
        sims = batch @ train_norm.T  # (batch_size, n_train)
        # Cosine distance
        dists = 1.0 - sims
        # Nearest neighbor
        nn_dists = dists.min(axis=1)
        nn_indices = dists.argmin(axis=1)
        distances.append(nn_dists)
        indices.append(nn_indices)

    distances = np.concatenate(distances)
    indices = np.concatenate(indices)

    return distances, indices


def compute_diversity(embeddings):
    """Compute average pairwise distance (diversity) in embedding space."""
    print("Computing diversity of generated images...")

    # Normalize
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Pairwise cosine distances (sample for efficiency)
    n = len(embeddings)
    sample_size = min(1000, n)

    if sample_size < n:
        indices = np.random.choice(n, sample_size, replace=False)
        sample = norm[indices]
    else:
        sample = norm

    sims = sample @ norm.T
    dists = 1.0 - sims

    # Exclude self-distances (diagonal)
    mask = np.eye(len(sample), n, dtype=bool) == False
    avg_diversity = dists[mask].mean()

    return avg_diversity


def analyze_and_report(distances, indices, gen_count, train_count, diversity, out_dir):
    """Analyze distances and generate report."""
    os.makedirs(out_dir, exist_ok=True)

    # Statistics
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    std_dist = distances.std()

    # Thresholds for memorization detection
    # Very small distances (< 0.1) in normalized cosine space suggest near-exact matches
    threshold_extreme = 0.05
    threshold_suspicious = 0.15

    n_extreme = (distances < threshold_extreme).sum()
    n_suspicious = (distances < threshold_suspicious).sum()
    pct_extreme = 100.0 * n_extreme / gen_count
    pct_suspicious = 100.0 * n_suspicious / gen_count

    # Conclusion logic
    if pct_extreme > 5.0:  # >5% with extremely small distances
        conclusion = "LIKELY MEMORIZATION"
        confidence = "HIGH"
    elif pct_suspicious > 20.0:  # >20% with suspicious distances
        conclusion = "LIKELY MEMORIZATION"
        confidence = "MEDIUM"
    elif pct_extreme > 1.0:
        conclusion = "UNCERTAIN (some suspiciously close matches)"
        confidence = "LOW"
    else:
        conclusion = "LIKELY NO MEMORIZATION"
        confidence = "HIGH"

    # Generate report
    report = f"""
{'='*70}
MEMORIZATION DETECTION REPORT
{'='*70}

DATASET:
  Generated images: {gen_count}
  Training images (reference): {train_count}

NEAREST-NEIGHBOR DISTANCE ANALYSIS (Cosine Distance in Embedding Space):
  Min distance:           {min_dist:.6f}
  Max distance:           {max_dist:.6f}
  Mean distance:          {mean_dist:.6f}
  Median distance:        {median_dist:.6f}
  Std dev:                {std_dist:.6f}

MEMORIZATION INDICATORS:
  Extremely close (< {threshold_extreme}):  {n_extreme:3d} / {gen_count:3d}  ({pct_extreme:5.2f}%)
  Suspicious (< {threshold_suspicious}):  {n_suspicious:3d} / {gen_count:3d}  ({pct_suspicious:5.2f}%)

DIVERSITY OF GENERATED IMAGES:
  Avg pairwise distance (diversity): {diversity:.6f}
  (Higher = more diverse; values typically in [0.3, 1.0])

CONCLUSION:
  {conclusion}
  Confidence: {confidence}

INTERPRETATION:
  - If many generated images have very small distances to training images,
    the model may be memorizing or overfitting.
  - A mean distance around 0.5 suggests reasonable diversity.
  - Mean distance < 0.3 with many suspicious matches suggests memorization.

{'='*70}
"""

    print(report)

    # Save report
    report_path = os.path.join(out_dir, 'memorization_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Histogram
    axes[0].hist(distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(threshold_extreme, color='red', linestyle='--', lw=2, label=f'Extreme threshold ({threshold_extreme})')
    axes[0].axvline(threshold_suspicious, color='orange', linestyle='--', lw=2, label=f'Suspicious threshold ({threshold_suspicious})')
    axes[0].axvline(mean_dist, color='green', linestyle='-', lw=2, label=f'Mean ({mean_dist:.3f})')
    axes[0].set_xlabel('Cosine Distance to Nearest Training Image', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Nearest-Neighbor Distance Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # CDF
    sorted_dist = np.sort(distances)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    axes[1].plot(sorted_dist, cdf, color='steelblue', lw=2)
    axes[1].axvline(threshold_extreme, color='red', linestyle='--', lw=2, label=f'Extreme threshold ({threshold_extreme})')
    axes[1].axvline(threshold_suspicious, color='orange', linestyle='--', lw=2, label=f'Suspicious threshold ({threshold_suspicious})')
    axes[1].set_xlabel('Cosine Distance to Nearest Training Image', fontsize=11)
    axes[1].set_ylabel('Cumulative Fraction', fontsize=11)
    axes[1].set_title('CDF of Nearest-Neighbor Distances', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'distance_histogram.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {plot_path}")

    return conclusion, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='Directory containing generated images')
    parser.add_argument('--out_dir', type=str, default='outputs/memorization',
                        help='Output directory for report and plots')
    parser.add_argument('--max_gen', type=int, default=None,
                        help='Max generated images to analyze (None=all)')
    parser.add_argument('--max_train', type=int, default=None,
                        help='Max training images to use as reference (None=all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # Load images
    train_images = load_images(args.train_dir, max_images=args.max_train, device=device)
    gen_images = load_images(args.gen_dir, max_images=args.max_gen, device=device)

    print(f"\nTrain shape: {train_images.shape}")
    print(f"Generated shape: {gen_images.shape}\n")

    # Get embeddings
    train_emb = get_inception_embeddings(train_images, device=device)
    gen_emb = get_inception_embeddings(gen_images, device=device)

    print(f"Train embeddings shape: {train_emb.shape}")
    print(f"Generated embeddings shape: {gen_emb.shape}\n")

    # Find nearest neighbors
    distances, indices = compute_nearest_neighbors(gen_emb, train_emb)

    # Compute diversity
    diversity = compute_diversity(gen_emb)

    # Analyze and report
    conclusion, confidence = analyze_and_report(
        distances, indices,
        len(gen_images), len(train_images),
        diversity,
        args.out_dir
    )


if __name__ == '__main__':
    main()
