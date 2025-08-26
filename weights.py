import numpy as np
m = np.array([5.0, 4.0, 2.0])
sd = np.array([5.0, 2.0, 1.0])
corr = np.identity(3)
Sigma = np.outer(sd, sd) * corr

def mean_variance_weights(m, Sigma, lam):
    invSigma = np.linalg.pinv(Sigma)
    w_star = (1.0/(2.0*lam)) * invSigma.dot(m)
    w_star[w_star < 0] = 0.0
    if w_star.sum() == 0:
        return np.ones_like(w_star) / len(w_star)
    return w_star / w_star.sum()

for lam in [0.01, 0.03, 0.07, 0.12]:
    w = mean_variance_weights(m, Sigma, lam)
    mu_p = w.dot(m)
    var_p = w.dot(Sigma).dot(w)
    print(f"lambda: {lam:.4g}  weights: {np.round(w,3)}  mean: {mu_p:.2f}  stdev: {np.sqrt(var_p):.2f}")
