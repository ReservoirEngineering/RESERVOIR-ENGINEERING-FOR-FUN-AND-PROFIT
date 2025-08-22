import numpy as np

m = np.array([120.0, 90.0, 70.0])  # EMV
sd = np.array([220.0, 140.0, 90.0])
corr = np.array([[1.0, 0.8, 0.5],
                 [0.8, 1.0, 0.2],
                 [0.5, 0.2, 1.0]])
Sigma = np.outer(sd, sd) * corr

def mean_variance_weights(m, Sigma, lam):
    invSigma = np.linalg.pinv(Sigma)
    w_star = (1.0/(2.0*lam)) * invSigma.dot(m)
    w_star[w_star < 0] = 0.0
    if w_star.sum() == 0:
        return np.ones_like(w_star) / len(w_star)
    return w_star / w_star.sum()

for lam in [1e-4, 2e-4, 5e-4]:
    w = mean_variance_weights(m, Sigma, lam)
    mu_p = w.dot(m)
    var_p = w.dot(Sigma).dot(w)
    print(lam, "weights:", w, "mean:", mu_p, "stdev:", np.sqrt(var_p))
