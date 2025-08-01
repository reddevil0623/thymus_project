#!/usr/bin/env python3
import sys
import os
import math
import gc
import random
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import ot

# Make sure Python can find your custom eucalc module
sys.path.append('eucalc_directory')
import eucalc as ec

# — USER SETTINGS —
input_dir      = "Segmentation_Masks_Ben/Old_vs_Young"
output_dir     = "subimages_from_square_200x200_tiff"
sampeuler_dir  = "sampeulers"
overlay_dir     = "overlays"

min_pixels     = 2000
threshold      = 0
margin         = 0
patch_w, patch_h = 200, 200

k              = 100
xpoints        = 3000
delta_x        = 3.0 / (xpoints - 1)
n_young        = 500         # number of young patches to sample
num_slices     = 50
desc_suffix    = f"_xp{xpoints}.npy"

# Number of plots to generate
m_plots        = 20    # <-- set this to the desired number of repeats

# reproducible sampling
random.seed(0)
np.random.seed(0)

# ensure directories exist
os.makedirs(output_dir,    exist_ok=True)
os.makedirs(sampeuler_dir, exist_ok=True)
os.makedirs(overlay_dir,    exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

# ─── Helpers ─────────────────────────────────────────────────────────────
def segment_to_square(fn, margin=0, threshold=0):
    path = os.path.join(input_dir, fn)
    img = Image.open(path)
    arr = np.array(img)
    mask = np.any(arr[..., :3] > threshold, axis=2) if arr.ndim == 3 else arr > threshold
    if not mask.any():
        raise ValueError(f"No pixels > {threshold} in {fn}")
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0, x0 = max(y0-margin, 0), max(x0-margin, 0)
    y1, x1 = min(y1+margin, arr.shape[0]-1), min(x1+margin, arr.shape[1]-1)
    h, w = y1-y0+1, x1-x0+1
    side = max(h, w)
    cy, cx = y0 + h//2, x0 + w//2
    y0s = max(cy-side//2, 0)
    x0s = max(cx-side//2, 0)
    y1s = min(y0s+side, arr.shape[0])
    x1s = min(x0s+side, arr.shape[1])
    cropped = arr[y0s:y1s, x0s:x1s]
    return Image.fromarray(cropped)

def sliced_wasserstein_l1(emp1, emp2, num_slices=100, delta_x=1.0):
    k, T = emp1.shape
    X, Y = emp1 * delta_x, emp2 * delta_x
    vals = np.empty(num_slices, dtype=np.float32)
    for i in range(num_slices):
        u = np.random.uniform(-1, 1, size=(T,))
        u /= np.max(np.abs(u))
        p, q = X.dot(u), Y.dot(u)
        p.sort(); q.sort()
        vals[i] = np.mean(np.abs(p - q))
    return vals.mean()

class SampEuler:
    def __init__(self, arr, k, xinterval, xpoints):
        self.arr, self.k, self.xinterval, self.xpoints = arr, k, xinterval, xpoints
    def compute(self):
        cplx = ec.EmbeddedComplex(self.arr)
        cplx.preproc_ect()
        thetas = np.random.uniform(0, 2*np.pi, self.k)
        T = np.linspace(self.xinterval[0], self.xinterval[1], self.xpoints)
        out = np.empty((self.k, self.xpoints), dtype=np.float32)
        for i, θ in enumerate(thetas):
            u = np.array((np.sin(θ), np.cos(θ)))
            ect_dir = cplx.compute_euler_characteristic_transform(u)
            out[i] = [ect_dir.evaluate(t) for t in T]
        return out

def ensure_patches_and_descriptors():
    # 1) Crop, PAD & tile
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif','.tiff'))]
    names_k8 = [f for f in all_files if 'K8' in f]

    existing = glob(os.path.join(output_dir, "*.tif"))
    if not existing:
        patch_names = []
        for fn in names_k8:
            sq = segment_to_square(fn, margin, threshold)
            arr = np.array(sq)
            H, W = arr.shape[:2]

            # pad bottom/right to multiples of patch size
            pad_h = (math.ceil(H/patch_h)*patch_h) - H
            pad_w = (math.ceil(W/patch_w)*patch_w) - W
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w)) + ((0,0),)*(arr.ndim-2),
                mode='constant', constant_values=0
            )

            H2, W2 = arr.shape[:2]
            nrows, ncols = H2 // patch_h, W2 // patch_w
            base = os.path.splitext(fn)[0]
            for r in range(nrows):
                for c in range(ncols):
                    y0, x0 = r*patch_h, c*patch_w
                    tile = arr[y0:y0+patch_h, x0:x0+patch_w]
                    if np.count_nonzero(tile > threshold) < min_pixels:
                        continue
                    rIdx = (nrows-1)-r
                    pname = f"{base}_sq_r{rIdx}_c{c}.tif"
                    Image.fromarray(tile).save(os.path.join(output_dir, pname))
                    patch_names.append(pname)
        print(f"Generated {len(patch_names)} patches.")
    else:
        patch_names = [os.path.basename(p) for p in existing]

    # 2) Compute descriptors if missing
    for fn in patch_names:
        path = os.path.join(sampeuler_dir, fn + desc_suffix)
        if not os.path.exists(path):
            arr  = np.array(Image.open(os.path.join(output_dir, fn)))
            desc = SampEuler(arr, k, (-1.5,1.5), xpoints).compute()
            np.save(path, desc)
            gc.collect()

    return patch_names

if __name__ == "__main__":
    # Prepare patches and descriptors
    patch_names = ensure_patches_and_descriptors()

    # split young vs old
    young = [n for n in patch_names if n.startswith('Y')]
    old   = [n for n in patch_names if n.startswith('O')]

    # Generate multiple plots
    for idx in range(m_plots):
        # pick one random old, and sample n_young young
        test_name   = random.choice(old)
        young_names = random.sample(young, min(n_young, len(young)))

        # load descriptors
        test_desc   = np.load(os.path.join(sampeuler_dir, test_name + desc_suffix))
        young_descs = [np.load(os.path.join(sampeuler_dir, y + desc_suffix)) for y in young_names]

        # combine for barycenter
        all_ms   = [test_desc] + young_descs
        n        = len(all_ms)
        weights  = [np.ones(m.shape[0]) / m.shape[0] for m in all_ms]
        W_uniform = np.ones(n) / n

        # pick initial support
        D = np.zeros((n, n), float)
        for i in range(n):
            for j in range(i+1, n):
                d = sliced_wasserstein_l1(all_ms[i], all_ms[j], num_slices, delta_x)
                D[i, j] = D[j, i] = d
        med = np.argmin(D.sum(axis=1))
        X0, b0 = all_ms[med], np.ones(all_ms[med].shape[0]) / all_ms[med].shape[0]

        # compute barycenter
        mu = ot.lp.free_support_barycenter(
            all_ms, weights, X0, b0,
            weights=W_uniform, numItermax=1000, stopThr=1e-6
        )

        # compute distances to barycenter
        S0  = sliced_wasserstein_l1(test_desc, mu, num_slices, delta_x)
        Sis = [sliced_wasserstein_l1(y, mu, num_slices, delta_x) for y in young_descs]

        # plot distribution
        plt.figure(figsize=(6, 4))
        plt.hist(Sis, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(S0, linestyle='--', linewidth=2,
                    label=f"Old ({test_name}): {S0:.3f}")
        plt.xlabel('Sliced–Wasserstein Distance to Barycenter')
        plt.ylabel('Count')
        plt.title(f'Young patches vs. selected Old patch (plot {idx+1})')
        plt.legend()
        out_png = os.path.join(overlay_dir, f"distance_dist_{test_name}_{idx+1}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved distribution plot {idx+1}/{m_plots}: {out_png}")
