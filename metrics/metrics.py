"""FID, KID, Inception Score via pretrained InceptionV3 (torchvision)."""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg


class InceptionMetrics:
    def __init__(self, device, batch_size=256):
        self.device = device
        self.batch_size = batch_size

        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.eval().to(device)

        self._pool, self._probs = [], []

        def _hook_pool(m, inp, out):
            self._pool.append(out.flatten(1).detach().cpu())   # (N,2048)

        def _hook_fc(m, inp, out):
            self._probs.append(torch.softmax(out, dim=1).detach().cpu())

        model.avgpool.register_forward_hook(_hook_pool)
        model.fc.register_forward_hook(_hook_fc)
        self.model = model

    @torch.no_grad()
    def get_activations(self, images_neg1_1):
        """images_neg1_1: Tensor (N,3,H,W) in [-1,1]. Returns (features, probs) numpy."""
        self._pool.clear(); self._probs.clear()
        imgs = (images_neg1_1.clamp(-1, 1) + 1) / 2   # [-1,1] -> [0,1]
        for i in range(0, len(imgs), self.batch_size):
            b = imgs[i:i+self.batch_size].to(self.device)
            b = F.interpolate(b, size=(299, 299), mode='bilinear', align_corners=False)
            self.model(b)
        feats = torch.cat(self._pool,  dim=0).numpy()   # (N, 2048)
        probs = torch.cat(self._probs, dim=0).numpy()   # (N, 1000)
        return feats, probs

    def compute_fid(self, mu_r, sigma_r, feats_f):
        mu_f = feats_f.mean(0)
        sigma_f = np.cov(feats_f, rowvar=False)
        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))

    def compute_kid(self, feats_r, feats_f, n_subsets=50, subset_size=1000):
        """Polynomial kernel MMD ×1000 (unbiased)."""
        d = feats_r.shape[1]
        n = min(len(feats_r), len(feats_f), subset_size)
        mmds = []
        for _ in range(n_subsets):
            r = feats_r[np.random.choice(len(feats_r), n, replace=False)]
            f = feats_f[np.random.choice(len(feats_f), n, replace=False)]
            k_rr = ((r @ r.T) / d + 1) ** 3
            k_ff = ((f @ f.T) / d + 1) ** 3
            k_rf = ((r @ f.T) / d + 1) ** 3
            mmd = ((k_rr.sum() - np.trace(k_rr)) / (n * (n - 1)) +
                   (k_ff.sum() - np.trace(k_ff)) / (n * (n - 1)) -
                   2 * k_rf.mean())
            mmds.append(mmd)
        return float(np.mean(mmds)) * 1000, float(np.std(mmds)) * 1000

    def compute_is(self, probs, n_splits=10):
        """Inception Score."""
        n, split = len(probs), len(probs) // n_splits
        scores = []
        for i in range(n_splits):
            p = probs[i*split:(i+1)*split]
            py = p.mean(0, keepdims=True)
            kl = p * (np.log(p + 1e-10) - np.log(py + 1e-10))
            scores.append(np.exp(kl.sum(1).mean()))
        return float(np.mean(scores)), float(np.std(scores))

    def compute_real_stats(self, dataloader):
        """Compute mu, sigma, all features on real dataset."""
        all_feats = []
        for imgs, _ in dataloader:
            feats, _ = self.get_activations(imgs)
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, 0)
        return all_feats.mean(0), np.cov(all_feats, rowvar=False), all_feats
