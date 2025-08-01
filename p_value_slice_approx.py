#!/usr/bin/env python3
import sys
import os
import gc
import re
import random
import json
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import ot
# Make sure Python can find your custom eucalc module
sys.path.append('eucalc_directory')
import eucalc as ec

# — USER SETTINGS —
input_dir        = "Segmentation_masks_Ben/Old_vs_Young"
output_dir       = "subimages_from_square_200x200_tiff"
sampeuler_dir    = "sampeulers"
min_pixels       = 2000
threshold        = 0
margin           = 0
patch_w, patch_h = 200, 200
k                = 100
xpoints          = 9000
# delta_x used for scaling the descriptor values for distance computation
delta_x          = 3.0 / (xpoints - 1)
n_young          = 100
# number of projection slices for sliced Wasserstein approximation
num_slices       = 100

# reproducible sampling
random.seed(0)
np.random.seed(0)

# Ensure directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sampeuler_dir, exist_ok=True)
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

# ─── Sliced Wasserstein under ℓ1 ground metric ────────────────────────────
def sliced_wasserstein_l1(emp1, emp2, num_slices=100, delta_x=1.0):
    """
    Approximate W1 distance under ℓ1 cost between two empirical measures.
    emp1, emp2: arrays of shape (k, T)
    """
    k, T = emp1.shape
    # scale descriptors
    X = emp1 * delta_x
    Y = emp2 * delta_x
    vals = np.empty(num_slices, dtype=np.float32)
    for i in range(num_slices):
        # sample from ℓ∞ unit sphere
        u = np.random.uniform(-1, 1, size=(T,))
        u /= np.max(np.abs(u))
        # project and sort
        p = X.dot(u)
        q = Y.dot(u)
        p.sort()
        q.sort()
        vals[i] = np.mean(np.abs(p - q))
    return vals.mean()

# ─── SampEuler descriptor ─────────────────────────────────────────────────
class SampEuler:
    def __init__(self, arr, k, xinterval, xpoints):
        self.arr, self.k, self.xinterval, self.xpoints = arr, k, xinterval, xpoints
    def compute(self):
        cplx = ec.EmbeddedComplex(self.arr)
        cplx.preproc_ect()
        thetas = np.random.uniform(0, 2*np.pi, self.k)
        T = np.linspace(self.xinterval[0], self.xinterval[1], self.xpoints)
        out = np.empty((self.k, self.xpoints), dtype=np.float32)
        for i in range(self.k):
            u = np.array((np.sin(thetas[i]), np.cos(thetas[i])))
            ect_dir = cplx.compute_euler_characteristic_transform(u)
            out[i] = [ect_dir.evaluate(t) for t in T]
        return out

# ─── 1) CROP & TILE ────────────────────────────────────────────────────────
all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif','.tiff'))]
names_k8 = [f for f in all_files if 'K8' in f]

if glob(os.path.join(output_dir, "*.tif")):
    patch_names = [os.path.basename(p) for p in sorted(glob(os.path.join(output_dir, "*.tif")))]
else:
    patch_names = []
    for fn in names_k8:
        sq = segment_to_square(fn, margin, threshold)
        arr = np.array(sq)
        H, W = arr.shape[:2]
        nrows, ncols = H // patch_h, W // patch_w
        base = os.path.splitext(fn)[0]
        for row in range(nrows):
            for col in range(ncols):
                y0, x0 = row*patch_h, col*patch_w
                tile = arr[y0:y0+patch_h, x0:x0+patch_w]
                if np.count_nonzero(tile > threshold) >= min_pixels:
                    rIdx = (nrows-1)-row
                    cIdx = col
                    name = f"{base}_sq_r{rIdx}_c{cIdx}.tif"
                    Image.fromarray(tile).save(os.path.join(output_dir, name))
                    patch_names.append(name)
    print(f"Generated {len(patch_names)} patches.")

# ─── 2) LOAD/COMPUTE descriptors & p-values ────────────────────────────────
pval_file = os.path.join(sampeuler_dir, "old_patch_pvalues_ellipse.json")
if os.path.exists(pval_file):
    with open(pval_file,'r') as f:
        pval_dict = json.load(f)
else:
    pval_dict = {}

# Phase 1: descriptors
print("Phase 1: computing descriptors…")
young = [n for n in patch_names if n.startswith('Y')]
old   = [n for n in patch_names if n.startswith('O')]
for fn in tqdm(patch_names, desc="Descriptors"):
    npy = os.path.join(sampeuler_dir, fn + '.npy')
    if not os.path.exists(npy):
        arr = np.array(Image.open(os.path.join(output_dir, fn)))
        np.save(npy, SampEuler(arr, k, (-1.5,1.5), xpoints).compute())
        gc.collect()

# Phase 2: p-values via sliced Wasserstein
print("Phase 2: computing p-values…")
todo = [n for n in old if n not in pval_dict]
for oname in tqdm(todo, desc="P-values"):
    test = np.load(os.path.join(sampeuler_dir, oname + '.npy'))
    sampled = random.sample(young, min(n_young, len(young)))
    yms = [np.load(os.path.join(sampeuler_dir, y + '.npy')) for y in sampled]
    all_ms = [test] + yms
    n = len(all_ms)
    D = np.zeros((n,n), dtype=float)
    for a in range(n):
        for b in range(a+1,n):
            D[a,b] = D[b,a] = sliced_wasserstein_l1(all_ms[a], all_ms[b], num_slices, delta_x)
    # barycenter & p-value
    weights = [np.ones(m.shape[0])/m.shape[0] for m in all_ms]
    med = np.argmin(D.sum(axis=1)); X0 = all_ms[med]
    b0 = np.ones(X0.shape[0])/X0.shape[0]; W = np.ones(n)/n
    mu = ot.lp.free_support_barycenter(all_ms, weights, X0, b0,
                                        weights=W, numItermax=1000, stopThr=1e-6)
    S0 = sliced_wasserstein_l1(test, mu, num_slices, delta_x)
    Sis = [sliced_wasserstein_l1(m, mu, num_slices, delta_x) for m in yms]
    pval_dict[oname] = float(np.mean([S0>=s for s in Sis]))
    with open(pval_file,'w') as f:
        json.dump(pval_dict,f,indent=2)
    gc.collect()

print(f"→ p-values saved to {pval_file}")

# ─── 3) VISUALIZE ─────────────────────────────────────────────────────────
for fn in names_k8:
    base = os.path.splitext(fn)[0]
    orig = np.array(Image.open(os.path.join(input_dir, fn)))
    H, W = orig.shape[:2]
    heat = np.zeros((H,W), dtype=np.float32)
    nrows, ncols = H//patch_h, W//patch_w
    for pname,pv in pval_dict.items():
        if not pname.startswith(base): continue
        r,c = map(int,re.search(r"_r(\d+)_c(\d+)",pname).groups())
        y0,x0 = (nrows-1-r)*patch_h, c*patch_w
        heat[y0:y0+patch_h, x0:x0+patch_w] = 1.0 - pv
    plt.figure(figsize=(6,6))
    plt.imshow(orig, cmap='gray')
    plt.imshow(heat, alpha=0.6, vmin=0, vmax=1)
    plt.title(f"{base} (1−p overlay)")
    plt.axis('off')
    plt.show()