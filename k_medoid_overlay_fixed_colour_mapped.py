#!/usr/bin/env python3
"""
Cluster square tissue‐patch masks into 5 groups using K‑medoids, then export results in
three parallel formats:

1. **Colour overlay PNGs**               → `clusterings_manual/`
2. **TIFF label matrices (square crop)** → `cluster_label_matrices/`
3. **NumPy label matrices (full raw size, ALIGN with original image)** → `cluster_label_npy/`

Change in this revision
-----------------------
* The `.npy` array is now **the same shape as the raw input image** (no crop, no
  padding).  We embed the square‑crop label matrix back into its original
  `(y0s:y0s+H, x0s:x0s+W)` window so `np.load()` aligns directly with the raw
  pixel coordinates that Lili uses.
"""
import os
import math
import json
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.colors as mcolors

# — USER SETTINGS —
d               = 5                    # number of clusters
random_seed     = 0
np.random.seed(random_seed)

# directories
input_dir       = "Segmentation_Masks_Ben/Old_vs_Young"
patch_dir       = "subimages_from_square_200x200_tiff_padded"
distance_path   = "distance_matrix.npy"
index_map_path  = "patch_index_map.json"
output_dir      = "clusterings_k_medoid_colour_mapped"             # coloured PNG overlays
label_dir       = "cluster_label_matrices"         # TIFF label matrices (square crop)
npy_dir         = "cluster_label_npy"              # NumPy arrays matching RAW size

# tiling / mask params
threshold       = 0
margin          = 0
patch_w, patch_h = 200, 200

# ensure output directories exist
for d_ in (output_dir, label_dir, npy_dir):
    os.makedirs(d_, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------
# 1) Load distance matrix & index mapping
# ----------------------------------------------------------------------
print("Loading distance matrix and index map…")
D = np.load(distance_path, mmap_mode="r")
with open(index_map_path, "r") as f:
    mapping = json.load(f)

i2name = {int(i): name for i, name in mapping["i2name"].items()}
n = D.shape[0]
print(f"Loaded D[{n}×{n}]")

# ----------------------------------------------------------------------
# 2) K‑medoids clustering (distance‑matrix mode)
# ----------------------------------------------------------------------
print(f"Running pyclustering K‑medoids with n_clusters = {d} …")
initial_medoids = list(np.random.choice(n, d, replace=False))
km = kmedoids(D, initial_medoids, data_type="distance_matrix", random_state=random_seed)
km.process()

labels = np.empty(n, dtype=int)
for cid, pts in enumerate(km.get_clusters()):
    labels[pts] = cid

# — PERMUTE CLUSTER INDICES (1→5, 2→4, 3→1, 4→2, 5→3)
perm = np.array([4, 3, 0, 1, 2], dtype=int)
labels = perm[labels]
cluster_map = {i2name[i]: int(labels[i]) for i in range(n)}  # 0‑based IDs

# ----------------------------------------------------------------------
# 3) Colormap helpers (jet)
# ----------------------------------------------------------------------
cmap_cont      = plt.get_cmap("jet")
tick_positions = np.linspace(0, 1, d)
tick_colors    = [cmap_cont(pos)[:3] for pos in tick_positions]
sm_cont        = plt.cm.ScalarMappable(cmap=cmap_cont, norm=mcolors.Normalize(vmin=0, vmax=1))
sm_cont.set_array([])

# ----------------------------------------------------------------------
# 4) Overlay clusters & write label matrices
# ----------------------------------------------------------------------
print("Generating overlays and label matrices…")
bases = [f for f in os.listdir(input_dir)
         if f.lower().endswith((".tif", ".tiff")) and "K8" in f]

for base_fn in tqdm(bases, desc="Processing"):
    base, _ = os.path.splitext(base_fn)

    # --- load original ---------------------------------------------------
    img  = Image.open(os.path.join(input_dir, base_fn))
    arr  = np.array(img)
    mask = (np.any(arr[..., :3] > threshold, axis=2)
            if arr.ndim == 3 else arr > threshold)
    ys, xs = np.where(mask)
    y0b, y1b = ys.min(), ys.max(); x0b, x1b = xs.min(), xs.max()
    y0 = max(y0b - margin, 0); x0 = max(x0b - margin, 0)
    y1 = min(y1b + margin, arr.shape[0]-1); x1 = min(x1b + margin, arr.shape[1]-1)
    h, w = y1 - y0 + 1, x1 - x0 + 1
    side = max(h, w)
    cy, cx = y0 + h//2, x0 + w//2
    y0s = max(cy - side//2, 0); x0s = max(cx - side//2, 0)
    sq  = arr[y0s:y0s+side, x0s:x0s+side]
    seg = np.array(Image.fromarray(sq).convert("L"))

    # --- pad so it divides into patches ---------------------------------
    H, W = seg.shape
    pad_h = (math.ceil(H/patch_h)*patch_h) - H
    pad_w = (math.ceil(W/patch_w)*patch_w) - W
    seg_pad  = np.pad(seg, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    mask_pad = seg_pad > threshold
    H2, W2   = seg_pad.shape
    nrows, ncols = H2 // patch_h, W2 // patch_w

    # --- canvases --------------------------------------------------------
    rgb_overlay = np.zeros((H2, W2, 3), dtype=float)
    label_pad   = np.zeros((H2, W2), dtype=np.uint16)   # for TIFF

    # --- paint every patch ----------------------------------------------
    for patch_name, cl in cluster_map.items():
        if not patch_name.startswith(base + "_"):
            continue
        r, cidx = map(int, re.search(r"_r(\d+)_c(\d+)", patch_name).groups())
        y0p = (nrows - 1 - r) * patch_h
        x0p = cidx * patch_w
        pm = mask_pad[y0p:y0p+patch_h, x0p:x0p+patch_w]

        color = cmap_cont(cl / (d - 1))[:3]
        rgb_overlay[y0p:y0p+patch_h, x0p:x0p+patch_w][pm] = color
        label_pad[y0p:y0p+patch_h, x0p:x0p+patch_w][pm] = np.uint16(cl + 1)

    # --- crop padding back (square) -------------------------------------
    label_mat = label_pad[:H, :W]

    # --- embed into full‑size canvas ------------------------------------
    Hraw, Wraw = arr.shape[:2]
    label_full = np.zeros((Hraw, Wraw), dtype=np.uint16)
    label_full[y0s:y0s+H, x0s:x0s+W] = label_mat

    # --- save TIFF (square) & NPY (full) --------------------------------
    Image.fromarray(label_mat).save(os.path.join(label_dir, f"{base}_cluster_labels.tif"))
    np.save(os.path.join(npy_dir, f"{base}_cluster_labels.npy"), label_full)

    # --- overlay PNG -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(W2/100, H2/100), dpi=100)
    fig.subplots_adjust(right=0.70)
    ax.imshow(seg_pad, cmap="gray")
    ax.imshow(rgb_overlay, alpha=0.6)
    ax.axis("off")

    cbar = fig.colorbar(sm_cont, ax=ax, fraction=0.046, pad=0.15)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"Cluster {i+1}" for i in range(d)])
    cbar.ax.tick_params(labelsize=50, rotation=0, pad=12, width=3, length=10)
    for tick, col in zip(cbar.ax.get_yticklabels(), tick_colors):
        tick.set_color(col)
        tick.set_fontweight("bold")

    fig.savefig(os.path.join(output_dir, f"{base}_clusters_jet_permuted.png"),
                dpi=100, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)

print("Finished.\n  Overlays PNG  ->", output_dir,
      "\n  TIFF (square) ->", label_dir,
      "\n  NPY (raw size)->", npy_dir)
