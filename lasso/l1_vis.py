import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. SETUP 2D PARAMETER SPACE
# ============================================================

beta1 = np.linspace(-3, 3, 400)
beta2 = np.linspace(-3, 3, 400)
B1, B2 = np.meshgrid(beta1, beta2)

# ============================================================
# 2. LOSS CONTOURS (ELLIPTICAL)
# Example quadratic loss: (β1 - 1)^2 + 0.5(β2 - 1)^2
# ============================================================

loss = (B1 - 1)**2 + 0.5 * (B2 - 1)**2

# ============================================================
# 3. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))

# Loss contours
ax.contour(B1, B2, loss, levels=8, colors="gray")

# L1 constraint: |β1| + |β2| = c
c = 1.5
ax.plot([c, 0, -c, 0, c], [0, c, 0, -c, 0],
        color="red", linewidth=2, label="L1 constraint")

# L2 constraint: β1^2 + β2^2 = c^2
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(c*np.cos(theta), c*np.sin(theta),
        color="blue", linewidth=2, label="L2 constraint")

# Axis lines
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")
ax.set_title("Geometry of L1 vs L2 Regularisation")
ax.legend()
ax.set_aspect("equal")

plt.show()
