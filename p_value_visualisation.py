#!/usr/bin/env python3
import sys
import os
import re
import random
import gc
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Topological & optimal transport
import ot
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# If your eucalc package lives in a custom dir:
sys.path.append('eucalc_directory')
import eucalc as ec

# — USER SETTINGS — 
input_dir       = "Segmentation_masks_Ben/Old_vs_Young"
output_dir      = "subimages_from_square_200x200_tiff"
sampeuler_dir   = "sampeulers"            # where to save .npy descriptors
min_pixels      = 2000
threshold       = 0                       # pixels > threshold count as foreground
margin          = 10                      # padding around mask before cropping square
patch_w, patch_h= 200, 200

# Memory‐control parameters:
k                = 100                    # reduce from 480
xpoints          = 2000                   # reduce from 9000
delta_x          = 3.0 / xpoints
n_young          = 100                     # sample this many young patches per test
random.seed(0)                           # reproducible sampling

# Ensure folders exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sampeuler_dir, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None             # disable PIL decompression bomb warning


# — 1) CROP EACH MASK → CENTERED SQUARE + TILE INTO PATCHES —
def segment_to_square(fn, margin=0, threshold=0):
    img = Image.open(os.path.join(input_dir, fn))
    arr = np.array(img)
    if arr.ndim == 3:
        mask = np.any(arr[..., :3] > threshold, axis=2)
    else:
        mask = arr > threshold
    if not mask.any():
        raise ValueError(f"No pixels > {threshold} in {fn}")
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, arr.shape[0] - 1)
    x1 = min(x1 + margin, arr.shape[1] - 1)
    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    side = max(h, w)
    cy, cx = y0 + h//2, x0 + w//2
    y0s = max(cy - side//2, 0)
    x0s = max(cx - side//2, 0)
    y1s = min(y0s + side, arr.shape[0])
    x1s = min(x0s + side, arr.shape[1])
    crop = arr[y0s:y1s, x0s:x1s]
    return Image.fromarray(crop)

# load or generate patches
all_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif','.tiff'))]
names_k8   = [f for f in all_files if 'K8' in f]

def load_patches():
    fps = sorted(glob(os.path.join(output_dir, "*.tif")))
    imgs, names = [], []
    for fp in fps:
        imgs.append(Image.open(fp))
        names.append(os.path.basename(fp))
    return imgs, names

if glob(os.path.join(output_dir, "*.tif")):
    patches, patch_names = load_patches()
else:
    patches, patch_names = [], []
    for fn in names_k8:
        sq = segment_to_square(fn, margin, threshold)
        arr = np.array(sq)
        H, W = arr.shape[:2]
        nrows, ncols = H//patch_h, W//patch_w
        base = os.path.splitext(fn)[0]
        for row in range(nrows):
            for col in range(ncols):
                y0, x0 = row*patch_h, col*patch_w
                tile = arr[y0:y0+patch_h, x0:x0+patch_w]
                if np.count_nonzero(tile > threshold) >= min_pixels:
                    # invert row index to match bottom‐up convention
                    rIdx = (nrows - 1) - row
                    cIdx = col
                    outn = f"{base}_sq_r{rIdx}_c{cIdx}.tif"
                    Image.fromarray(tile).save(os.path.join(output_dir, outn))
                    patches.append(Image.fromarray(tile))
                    patch_names.append(outn)
    print(f"Generated and loaded {len(patches)} patches.")


# — 2) DEFINE DISTANCE & TEST FUNCTIONS —
def wasserstein_distance(emp1, emp2, p=2, delta_x=1.0):
    C = cdist(emp1, emp2, metric='minkowski', p=p) * (delta_x**(1.0/p))
    i, j = linear_sum_assignment(C)
    return np.mean(C[i,j]**p)**(1.0/p)

def monte_carlo_test(test_m, sampled_ms, p=2, delta_x=1.0):
    all_ms = [test_m] + sampled_ms
    weights = [np.ones(m.shape[0],dtype=float)/m.shape[0] for m in all_ms]
    N = len(all_ms)
    W = np.ones(N, dtype=float)/N
    k_dim = all_ms[0].shape[0]
    D = np.zeros((N, N), dtype=float)
    for a in range(N):
        for b in range(a+1, N):
            d = wasserstein_distance(all_ms[a], all_ms[b], p, delta_x)
            D[a,b] = D[b,a] = d
    med = np.argmin(D.sum(axis=1))
    X0 = all_ms[med].copy()
    b0 = np.ones(k_dim, dtype=float)/k_dim
    mu = ot.lp.free_support_barycenter(all_ms, weights, X0, b0,
                                        weights=W, numItermax=1000, stopThr=1e-6, verbose=False)
    S0 = wasserstein_distance(test_m, mu, p, delta_x)
    Sis = [wasserstein_distance(m, mu, p, delta_x) for m in sampled_ms]
    p_val = np.mean([S0 >= s for s in Sis])
    return S0, Sis, p_val


# — 3) DEFINE SampEuler CLASS — 
class SampEuler:
    def __init__(self, img_arr, k, xinterval, xpoints):
        self.arr = img_arr
        self.k = k
        self.xinterval = xinterval
        self.xpoints = xpoints

    def compute(self):
        cplx = ec.EmbeddedComplex(self.arr)
        cplx.preproc_ect()
        thetas = np.random.uniform(0, 2*np.pi, self.k+1)
        out = np.empty((self.k, self.xpoints), dtype=np.float32)
        T = np.linspace(self.xinterval[0], self.xinterval[1], self.xpoints)
        for i in range(self.k):
            direction = np.array((np.sin(thetas[i]), np.cos(thetas[i])))
            ect_dir = cplx.compute_euler_characteristic_transform(direction)
            out[i] = [ect_dir.evaluate(t) for t in T]
        return out


# — 4) PHASE 1: COMPUTE & SAVE SampEULER ARRAYS — 
print("Phase 1: computing SampEuler descriptors…")
all_patch_names = patch_names[:]  # capture order
for name in all_patch_names:
    npy_path = os.path.join(sampeuler_dir, name + ".npy")
    if os.path.exists(npy_path):
        continue
    img_arr = np.array(Image.open(os.path.join(output_dir, name)))
    ect = SampEuler(img_arr, k, (-1.5,1.5), xpoints).compute()
    np.save(npy_path, ect)
    del img_arr, ect
    gc.collect()
print("→ All descriptors saved to disk.")


# — 5) PHASE 2: COMPUTE p-VALUES WITH SAMPLING — 
print(f"Phase 2: Monte Carlo p-values (sampling {n_young} young patches each)…")
old_names   = [n for n in all_patch_names if n.startswith('O')]
young_names = [n for n in all_patch_names if n.startswith('Y')]
pval_dict   = {}

for oname in old_names:
    test_ect = np.load(os.path.join(sampeuler_dir, oname + ".npy"))

    sampled_y = random.sample(young_names, min(n_young, len(young_names)))
    young_ects = [np.load(os.path.join(sampeuler_dir, y + ".npy")) for y in sampled_y]

    _, _, pv = monte_carlo_test(test_ect, young_ects, p=2, delta_x=delta_x)
    pval_dict[oname] = pv

    del test_ect, young_ects
    gc.collect()

print("→ p-values computed for each old patch.")


# — 6) (Optional) SAVE p-values FOR LATER — 
import json
with open("old_patch_pvalues.json", "w") as f:
    json.dump(pval_dict, f, indent=2)
print("→ Saved old_patch_pvalues.json")


# — 7) OVERLAY HEATMAP ON ORIGINALS — 
for fn in names_k8:
    base = os.path.splitext(fn)[0]
    orig = np.array(Image.open(os.path.join(input_dir, fn)))
    H, W = orig.shape[:2]
    heat = np.zeros((H, W), dtype=np.float32)
    nrows, ncols = H//patch_h, W//patch_w

    for pname, pv in pval_dict.items():
        if not pname.startswith(base):
            continue
        m = re.search(r"_r(\d+)_c(\d+)\.tif$", pname)
        rIdx, cIdx = int(m.group(1)), int(m.group(2))
        intensity = 1.0 - pv
        y0 = (nrows - 1 - rIdx) * patch_h
        x0 = cIdx * patch_w
        heat[y0:y0+patch_h, x0:x0+patch_w] = intensity

    plt.figure(figsize=(6,6))
    plt.imshow(orig, cmap='gray')
    plt.imshow(heat, alpha=0.6, vmin=0, vmax=1)
    plt.title(f"{base} (1−p overlay)")
    plt.axis('off')
    plt.show()
