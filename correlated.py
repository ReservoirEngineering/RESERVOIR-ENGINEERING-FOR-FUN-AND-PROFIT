import numpy as np
from scipy.stats import norm, lognorm, beta
import matplotlib.pyplot as plt

# Parameters
N = 10000
rho = 0.6
mean = [0, 0]
cov = [[1, rho], [rho, 1]]

# Step 1: Correlated normals
Z = np.random.multivariate_normal(mean, cov, N)
# Step 2: Uniforms
U = norm.cdf(Z)
# Step 3: Map to marginals
mu_x, sigma_x = 0, 0.2  # lognormal parameters
a, b = 2, 5              # beta parameters

phi = lognorm.ppf(U[:,0], s=sigma_x, scale=np.exp(mu_x))
So = beta.ppf(U[:,1], a, b)

# Print sample output
print("First 5 phi:", phi[:5])
print("First 5 So:", So[:5])

# Optional: quick scatter plot to visualize joint samples
plt.scatter(phi, So, s=2, alpha=0.4)
plt.xlabel("phi (lognormal)")
plt.ylabel("So (beta)")
plt.title("Correlated random samples")
plt.show()
