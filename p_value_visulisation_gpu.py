#!/usr/bin/env python3
import sys, os, gc, re, random
from glob import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# GPU + progress bar imports
from numba import cuda
from tqdm import trange
from scipy.optimize import linear_sum_assignment
import ot
# custom ECT
sys.path.append('eucalc_directory')
import eucalc as ec

# — USER SETTINGS —
input_dir        = "Segmentation_masks_Ben/Old_vs_Young"
output_dir       = "subimages_from_square_200x200_tiff"
sampeuler_dir    = "sampeulers"
min_pixels       = 2000
threshold        = 0
margin           = 10
patch_w, patch_h = 200, 200
k                = 100
xpoints          = 2000
delta_x          = 3.0 / (xpoints - 1)
n_young          = 100
random.seed(0)

# ensure directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sampeuler_dir, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

# — 1) CROP & TILE —
def segment_to_square(fn, margin=0, threshold=0):
    img = Image.open(os.path.join(input_dir, fn))
    arr = np.array(img)
    mask = np.any(arr[..., :3] > threshold, axis=2) if arr.ndim==3 else arr>threshold
    if not mask.any():
        raise ValueError(f"No pixels > {threshold} in {fn}")
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    y0, x0 = max(y0-margin,0), max(x0-margin,0)
    y1, x1 = min(y1+margin,arr.shape[0]-1), min(x1+margin,arr.shape[1]-1)
    h, w = y1-y0+1, x1-x0+1; side = max(h,w)
    cy, cx = y0 + h//2, x0 + w//2
    y0s, x0s = max(cy-side//2,0), max(cx-side//2,0)
    y1s = min(y0s+side,arr.shape[0]); x1s = min(x0s+side,arr.shape[1])
    return Image.fromarray(arr[y0s:y1s, x0s:x1s])

all_files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.tif','.tiff'))]
names_k8 = [f for f in all_files if 'K8' in f]

# load or generate patches
if glob(os.path.join(output_dir, "*.tif")):
    patch_names = [os.path.basename(p) for p in sorted(glob(os.path.join(output_dir, "*.tif")))]
else:
    patch_names = []
    for fn in names_k8:
        sq = segment_to_square(fn, margin, threshold)
        arr = np.array(sq); H, W = arr.shape[:2]
        nrows, ncols = H//patch_h, W//patch_w
        base = os.path.splitext(fn)[0]
        for row in range(nrows):
            for col in range(ncols):
                y0, x0 = row*patch_h, col*patch_w
                tile = arr[y0:y0+patch_h, x0:x0+patch_w]
                if np.count_nonzero(tile>threshold)>=min_pixels:
                    rIdx = (nrows-1)-row; cIdx = col
                    outn = f"{base}_sq_r{rIdx}_c{cIdx}.tif"
                    Image.fromarray(tile).save(os.path.join(output_dir,outn))
                    patch_names.append(outn)
    print(f"Generated {len(patch_names)} patches.")

# — 2) GPU‐accelerated Wasserstein —
@cuda.jit
def cost_kernel(a, b, C, delta_x):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < b.shape[0]:
        s = 0.0
        for t in range(a.shape[1]):
            s += abs(a[i, t] - b[j, t])
        C[i, j] = s * delta_x

def exact_wasserstein(emp1, emp2, delta_x=1.0):
    a_dev = cuda.to_device(emp1.astype(np.float32))
    b_dev = cuda.to_device(emp2.astype(np.float32))
    N = emp1.shape[0]
    C_dev = cuda.device_array((N, N), dtype=np.float32)
    threads = (16, 16)
    blocks = ((N + threads[0]-1)//threads[0], (N + threads[1]-1)//threads[1])
    cost_kernel[blocks, threads](a_dev, b_dev, C_dev, delta_x)
    C = C_dev.copy_to_host()
    row, col = linear_sum_assignment(C)
    return np.mean(C[row, col])

# identical progress‐bar GPU distance matrix
def compute_distance_matrix_gpu(ects, delta_x):
    N = len(ects)
    D = np.zeros((N, N), dtype=float)
    for i in trange(N, desc="Wasserstein rows"):
        for j in range(i+1, N):
            D[i, j] = D[j, i] = exact_wasserstein(ects[i], ects[j], delta_x)
    return D

# — 3) SampEuler & P-values —
class SampEuler:
    def __init__(self, arr, k, xinterval, xpoints):
        self.arr, self.k, self.xinterval, self.xpoints = arr, k, xinterval, xpoints
    def compute(self):
        cplx = ec.EmbeddedComplex(self.arr);
        cplx.preproc_ect()
        thetas = np.random.uniform(0,2*np.pi,self.k)
        T = np.linspace(self.xinterval[0],self.xinterval[1],self.xpoints)
        out = np.empty((self.k,self.xpoints),dtype=np.float32)
        for i in range(self.k):
            direction = np.array((np.sin(thetas[i]),np.cos(thetas[i])))
            ect_dir = cplx.compute_euler_characteristic_transform(direction)
            out[i] = [ect_dir.evaluate(t) for t in T]
        return out

# Phase 1: descriptors
print("Phase 1: computing descriptors in parallel…")
def _compute_desc(fn):
    path = os.path.join(output_dir, fn)
    arr = np.array(Image.open(path))
    np.save(os.path.join(sampeuler_dir, fn+'.npy'), SampEuler(arr,k,(-1.5,1.5),xpoints).compute())
    gc.collect()
Parallel(n_jobs=-1, verbose=5)(delayed(_compute_desc)(fn) for fn in patch_names)

# Phase 2: p-values
young = [n for n in patch_names if n.startswith('Y')]
old   = [n for n in patch_names if n.startswith('O')]
pval_dict = {}
print("Phase 2: computing p-values…")
for oname in old:
    test = np.load(os.path.join(sampeuler_dir,oname+'.npy'))
    sampled = random.sample(young, min(n_young,len(young)))
    yms = [np.load(os.path.join(sampeuler_dir,y+'.npy')) for y in sampled]
    # GPU‐accelerated distance matrix with progress bar
    all_ms = [test] + yms
    D = compute_distance_matrix_gpu(all_ms, delta_x)
    weights = [np.ones(m.shape[0])/m.shape[0] for m in all_ms]
    med = np.argmin(D.sum(axis=1)); X0 = all_ms[med]
    b0 = np.ones(X0.shape[0])/X0.shape[0]; W = np.ones(len(all_ms))/len(all_ms)
    mu = ot.lp.free_support_barycenter(all_ms, weights, X0, b0,
                                        weights=W, numItermax=1000, stopThr=1e-6)
    S0 = exact_wasserstein(test, mu, delta_x)
    Sis = [exact_wasserstein(m, mu, delta_x) for m in yms]
    pval_dict[oname] = np.mean([S0>=s for s in Sis])
    gc.collect()

# Save and visualize
import json
json.dump(pval_dict, open("old_patch_pvalues.json","w"), indent=2)
print("Done: p-values saved.")
for fn in names_k8:
    base = os.path.splitext(fn)[0]
    orig = np.array(Image.open(os.path.join(input_dir,fn)))
    H,W = orig.shape[:2]; heat = np.zeros((H,W),dtype=np.float32)
    nrows,ncols = H//patch_h,W//patch_w
    for pname,pv in pval_dict.items():
        if not pname.startswith(base): continue
        r,c = map(int,re.search(r"_r(\d+)_c(\d+)",pname).groups())
        y0,x0 = (nrows-1-r)*patch_h, c*patch_w
        heat[y0:y0+patch_h,x0:x0+patch_w] = 1-pv
    plt.figure(figsize=(6,6))
    plt.imshow(orig,cmap='gray'); plt.imshow(heat,alpha=0.6,vmin=0,vmax=1)
    plt.title(f"{base} (1-p overlay)"); plt.axis('off'); plt.show()
