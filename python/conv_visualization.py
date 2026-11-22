"""
Task 7: 2D and 3D convolution visualization for FashionVision AI.
Needs only numpy + matplotlib. Run from the project root:
    python python/conv_visualization.py
Outputs: results/conv2d_steps.png, results/conv2d_kernels.png, results/conv3d_volume.png
Uses a real Fashion-MNIST image if dataset/fashion-mnist/t10k-images-idx3-ubyte exists.
"""
import os, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = "results"; os.makedirs(OUT, exist_ok=True)

def load_image(idx=0):
    path = "dataset/fashion-mnist/t10k-images-idx3-ubyte"
    if os.path.exists(path):
        with open(path, "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
        return data[idx].astype(float) / 255.0
    # fallback: synthetic 28x28 "shirt-like" shape
    img = np.zeros((28, 28)); img[6:24, 8:20] = 1; img[6:12, 3:25] = 1
    return img

def conv2d(x, k):
    """Valid cross-correlation (what CNN layers compute), written by hand."""
    kh, kw = k.shape
    H, W = x.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i+kh, j:j+kw] * k)
    return out

KERNELS = {
    "Sobel X (vertical edges)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float),
    "Sobel Y (horizontal edges)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float),
    "Blur (3x3 mean)": np.ones((3,3)) / 9.0,
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], float),
}

# ---------- Figure 1: step-by-step 2D convolution ----------
img = load_image(0)
k = KERNELS["Sobel X (vertical edges)"]
out = conv2d(img, k)
positions = [(4, 4), (10, 12), (18, 8)]
fig, axes = plt.subplots(len(positions), 3, figsize=(10, 9))
for r, (i, j) in enumerate(positions):
    patch = img[i:i+3, j:j+3]
    axes[r, 0].imshow(img, cmap="gray")
    axes[r, 0].add_patch(Rectangle((j-0.5, i-0.5), 3, 3, fill=False, ec="red", lw=2))
    axes[r, 0].set_title(f"Input 28x28, window at row {i}, col {j}")
    axes[r, 1].imshow(patch, cmap="gray", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            axes[r, 1].text(b, a, f"{patch[a,b]:.2f}\n×{k[a,b]:.0f}", ha="center", va="center", color="tomato", fontsize=8)
    axes[r, 1].set_title(f"Patch × kernel = {np.sum(patch*k):.2f}")
    axes[r, 2].imshow(out, cmap="RdBu_r")
    axes[r, 2].plot(j, i, "ks", mfc="none", ms=8)
    axes[r, 2].set_title(f"Output 26x26, value at ({i},{j})")
    for a in axes[r]: a.set_xticks([]); a.set_yticks([])
fig.suptitle("2D convolution: slide a 3x3 kernel, multiply, sum", fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/conv2d_steps.png", dpi=150); plt.close(fig)

# ---------- Figure 2: effect of different kernels ----------
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input")
for ax, (name, kern) in zip(axes[1:], KERNELS.items()):
    ax.imshow(conv2d(img, kern), cmap="gray"); ax.set_title(name, fontsize=9)
for a in axes: a.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/conv2d_kernels.png", dpi=150); plt.close(fig)

# ---------- Figure 3: 3D convolution over a multi-channel volume ----------
# Input volume: 3 channels (original + 2 edge maps), kernel is 3x3xC and spans ALL channels.
sx = conv2d(img, KERNELS["Sobel X (vertical edges)"])
sy = conv2d(img, KERNELS["Sobel Y (horizontal edges)"])
vol = np.stack([img[1:-1,1:-1], sx, sy], axis=0)      # shape (C, 26, 26)
C = vol.shape[0]
rng = np.random.default_rng(0)
K3 = rng.normal(0, 1, (C, 3, 3))                      # one 3D kernel (C x 3 x 3)
out3 = np.zeros((24, 24))
for i in range(24):
    for j in range(24):
        out3[i, j] = np.sum(vol[:, i:i+3, j:j+3] * K3)   # sums over channels too

def box(ax, x0, y0, z0, dx, dy, dz, color, alpha):
    v = [[x0,y0,z0],[x0+dx,y0,z0],[x0+dx,y0+dy,z0],[x0,y0+dy,z0],
         [x0,y0,z0+dz],[x0+dx,y0,z0+dz],[x0+dx,y0+dy,z0+dz],[x0,y0+dy,z0+dz]]
    faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
             [v[2],v[3],v[7],v[6]],[v[1],v[2],v[6],v[5]],[v[0],v[3],v[7],v[4]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors="k", alpha=alpha, linewidths=0.6))

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for c in range(C):                                    # three stacked channel planes
    box(ax, 0, 0, c*1.5, 10, 10, 0.15, ["#8ecae6","#90be6d","#f9c74f"][c], 0.35)
box(ax, 3, 3, -0.1, 3, 3, 1.5*(C-1)+0.35, "#e63946", 0.55)   # kernel cube spans all channels
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_zlim(-0.5, 4)
ax.set_title("Input volume (3 channels) and one 3×3×3 kernel\nkernel spans every channel")
ax.set_xlabel("width"); ax.set_ylabel("height"); ax.set_zlabel("channel"); ax.view_init(22, -60)
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(out3, cmap="viridis"); ax2.set_title("Output: ONE 2D feature map\n(sum over channels and the 3×3 window)")
ax2.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/conv3d_volume.png", dpi=150); plt.close(fig)

# ---------- Sanity check against PyTorch if available ----------
try:
    import torch, torch.nn.functional as F
    ref = F.conv2d(torch.tensor(img)[None,None].float(), torch.tensor(k)[None,None].float())[0,0].numpy()
    print("Hand-written conv2d matches PyTorch:", np.allclose(ref, out, atol=1e-4))
except ImportError:
    pass
print("Saved 3 figures in", OUT)