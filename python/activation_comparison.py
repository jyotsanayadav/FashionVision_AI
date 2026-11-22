"""
Task 9: compare step, sigmoid and tanh activations and their derivatives.
Run from the project root:  python python/activation_comparison.py
Output: results/activation_comparison.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
x = np.linspace(-6, 6, 1200)

step = (x >= 0).astype(float)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

d_step = np.zeros_like(x)            # zero everywhere (undefined at 0): no gradient to learn from
d_sigmoid = sigmoid * (1 - sigmoid)  # max 0.25
d_tanh = 1 - tanh ** 2               # max 1.0

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].plot(x, step, label="Step (McCulloch-Pitts)", color="#e63946", lw=2)
ax[0].plot(x, sigmoid, label="Sigmoid", color="#3a86ff", lw=2)
ax[0].plot(x, tanh, label="Tanh", color="#2e9e6b", lw=2)
ax[0].set_title("Activation functions")
ax[0].set_xlabel("input z"); ax[0].set_ylabel("output")

ax[1].plot(x, d_step, label="Step: derivative = 0", color="#e63946", lw=2)
ax[1].plot(x, d_sigmoid, label="Sigmoid (max 0.25)", color="#3a86ff", lw=2)
ax[1].plot(x, d_tanh, label="Tanh (max 1.0)", color="#2e9e6b", lw=2)
ax[1].set_title("Derivatives (what backpropagation uses)")
ax[1].set_xlabel("input z"); ax[1].set_ylabel("derivative")

for a in ax:
    a.axhline(0, color="grey", lw=0.6); a.axvline(0, color="grey", lw=0.6)
    a.grid(alpha=0.3); a.legend(loc="center right", fontsize=9)
fig.tight_layout()
fig.savefig("results/activation_comparison.png", dpi=150)

print("Max derivative -> sigmoid: %.3f | tanh: %.3f | step: %.3f" % (d_sigmoid.max(), d_tanh.max(), d_step.max()))
print("Saved results/activation_comparison.png")