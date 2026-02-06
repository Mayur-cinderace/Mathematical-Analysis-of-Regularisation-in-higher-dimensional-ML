import numpy as np
import matplotlib.pyplot as plt

beta1 = np.linspace(-3, 3, 400)
beta2 = np.linspace(-3, 3, 400)
B1, B2 = np.meshgrid(beta1, beta2)


loss = (B1 - 1)**2 + 0.5 * (B2 - 1)**2


fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(B1, B2, loss, levels=8, colors="gray")

c = 1.5
ax.plot([c, 0, -c, 0, c], [0, c, 0, -c, 0],
        color="red", linewidth=2, label="L1 constraint")

theta = np.linspace(0, 2*np.pi, 400)
ax.plot(c*np.cos(theta), c*np.sin(theta),
        color="blue", linewidth=2, label="L2 constraint")

ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")
ax.set_title("Geometry of L1 vs L2 Regularisation")
ax.legend()
ax.set_aspect("equal")

plt.show()
