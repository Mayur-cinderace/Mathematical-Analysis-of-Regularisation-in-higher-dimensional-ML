import numpy as np
import time
import tracemalloc
import pandas as pd

# -------------------------------
# NumPy implementations
# -------------------------------

def ridge_primal(X, y, lam):
    XtX = X.T @ X
    w = np.linalg.solve(XtX + lam * np.eye(X.shape[1]), X.T @ y)
    return w

def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

def lasso_cd(X, y, lam, T=50):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(T):
        for j in range(d):
            r = y - (X @ w) + X[:, j] * w[j]
            z = X[:, j].T @ r / n
            w[j] = soft_threshold(z, lam)
    return w

def elastic_net_cd(X, y, lam1, lam2, T=50):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(T):
        for j in range(d):
            r = y - (X @ w) + X[:, j] * w[j]
            z = X[:, j].T @ r / n
            w[j] = soft_threshold(z, lam1) / (1 + lam2)
    return w

# -------------------------------
# Experiment setup
# -------------------------------

np.random.seed(0)

n = 100                       # fixed samples
d_values = [500, 1000, 2000, 4000, 8000]
lam = 0.1
T = 30

records = []

# -------------------------------
# Benchmark loop
# -------------------------------

for d in d_values:
    X = np.random.randn(n, d)
    y = np.random.randn(n)

    # ---- Ridge ----
    tracemalloc.start()
    start = time.time()
    ridge_primal(X, y, lam)
    ridge_time = time.time() - start
    ridge_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # ---- Lasso ----
    tracemalloc.start()
    start = time.time()
    lasso_cd(X, y, lam, T)
    lasso_time = time.time() - start
    lasso_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # ---- Elastic Net ----
    tracemalloc.start()
    start = time.time()
    elastic_net_cd(X, y, lam, lam, T)
    enet_time = time.time() - start
    enet_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # ---- Store results ----
    records.append({
        "d (features)": d,
        "Ridge Time (s)": ridge_time,
        "Ridge Space (MB)": ridge_mem / 1e6,
        "Lasso Time (s)": lasso_time,
        "Lasso Space (MB)": lasso_mem / 1e6,
        "ElasticNet Time (s)": enet_time,
        "ElasticNet Space (MB)": enet_mem / 1e6,
    })

# -------------------------------
# Results table
# -------------------------------

df = pd.DataFrame(records)
print(df)

# -------------------------------
# Save results
# -------------------------------

df.to_csv("complexity_results.csv", index=False)
df.to_excel("complexity_results.xlsx", index=False)

print("\nSaved results to:")
print(" - complexity_results.csv")
print(" - complexity_results.xlsx")

import matplotlib.pyplot as plt

# -------------------------------
# Time complexity plot
# -------------------------------

plt.figure()
plt.plot(df["d (features)"], df["Ridge Time (s)"], marker='o', label="Ridge (Primal)")
plt.plot(df["d (features)"], df["Lasso Time (s)"], marker='o', label="Lasso")
plt.plot(df["d (features)"], df["ElasticNet Time (s)"], marker='o', label="Elastic Net")

plt.xlabel("Number of features (d)")
plt.ylabel("Time (seconds)")
plt.title("Time Complexity vs Feature Dimension")
plt.legend()
plt.grid(True)

plt.savefig("time_complexity_vs_d.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------
# Space complexity plot
# -------------------------------

plt.figure()
plt.plot(df["d (features)"], df["Ridge Space (MB)"], marker='o', label="Ridge (Primal)")
plt.plot(df["d (features)"], df["Lasso Space (MB)"], marker='o', label="Lasso")
plt.plot(df["d (features)"], df["ElasticNet Space (MB)"], marker='o', label="Elastic Net")

plt.xlabel("Number of features (d)")
plt.ylabel("Peak Memory (MB)")
plt.title("Space Complexity vs Feature Dimension")
plt.legend()
plt.grid(True)

plt.savefig("space_complexity_vs_d.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------
# Log–log time plot
# -------------------------------

plt.figure()
plt.loglog(df["d (features)"], df["Ridge Time (s)"], marker='o', label="Ridge (Primal)")
plt.loglog(df["d (features)"], df["Lasso Time (s)"], marker='o', label="Lasso")
plt.loglog(df["d (features)"], df["ElasticNet Time (s)"], marker='o', label="Elastic Net")

plt.xlabel("log(d)")
plt.ylabel("log(Time)")
plt.title("Log–Log Time Scaling")
plt.legend()
plt.grid(True, which="both")

plt.savefig("loglog_time_scaling.png", dpi=300, bbox_inches="tight")
plt.show()
