import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Number of simulations
Nsim = 10000

# Input distributions
A = np.random.uniform(800, 1200, Nsim)                 # acres
h = np.random.triangular(10, 15, 25, Nsim)              # ft
phi = np.random.normal(0.21, 0.025, Nsim)               # porosity
So = np.random.normal(0.75, 0.05, Nsim)                 # oil saturation
Bo = np.random.triangular(1.1, 1.2, 1.3, Nsim)          # formation volume factor

# Calculate OIP (in stock tank barrels)
N_OIP = 7758 * (A * h * phi * So) / Bo

# Sort for CDF
N_sorted = np.sort(N_OIP)
cdf = np.arange(1, Nsim+1) / Nsim

# Kernel Density Estimate for PDF
kde = gaussian_kde(N_OIP)
x_vals = np.linspace(N_OIP.min(), N_OIP.max(), 500)
pdf_vals = kde(x_vals)

# --- Plot PDF ---
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.hist(N_OIP, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Histogram')
plt.plot(x_vals, pdf_vals, color='darkorange', lw=2, label='KDE')
plt.title('PDF of OIP', fontsize=14)
plt.xlabel('OIP (STB)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# --- Plot CDF ---
plt.subplot(1, 2, 2)
plt.plot(N_sorted, cdf, color='darkred', lw=2)
plt.title('CDF of OIP', fontsize=14)
plt.xlabel('OIP (STB)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()