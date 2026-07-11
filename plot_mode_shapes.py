"""Quick mode shape visualizer. Usage: python plot_mode_shapes.py [npz_path] [sample_idx]"""
import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/sweep_20260709_220248/mode_shapes.npz"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

d = np.load(path)
shapes = d["mode_shapes"]  # (n, n_modes, ny, nx)
print(f"Loaded: {shapes.shape}, sample {idx}")

fig, axes = plt.subplots(1, shapes.shape[1], figsize=(4 * shapes.shape[1], 4))
if shapes.shape[1] == 1:
    axes = [axes]

for m, ax in enumerate(axes):
    img = shapes[idx, m]
    v = max(abs(img.min()), abs(img.max()))
    im = ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
    ax.set_title(f"Mode {m+1}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f"Sample {idx}")
plt.tight_layout()
plt.savefig("mode_shapes_preview.png", dpi=120)
print("Saved mode_shapes_preview.png")
plt.close()
