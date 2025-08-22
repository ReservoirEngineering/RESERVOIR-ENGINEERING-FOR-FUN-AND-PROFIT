import numpy as np
rng = np.random.default_rng(42)

def sample_lognormal_from_p50_p1090(p50, p10_to_p90):
    ratio = p10_to_p90
    sigma = np.log(ratio) / 2.563
    mu = np.log(p50)
    return np.exp(rng.normal(mu, sigma))

def simulate_project(n=20000, a=1.5e-9):
    disc = 0.10
    capex = 350.0
    lifcost = 12.0

    npvs = []
    utils = []
    for _ in range(n):
        price = np.exp(rng.normal(np.log(70.0), 0.30))
        reserves = sample_lognormal_from_p50_p1090(20.0, 2.0)
        plateau_bbl = 0.8 * reserves
        annual_bbl = plateau_bbl / 5.0
        annual_margin = max(price - lifcost, 0.0) * annual_bbl
        cashflows = [-capex] + [annual_margin] * 5
        npv = sum(cf / ((1 + disc)**t) for t, cf in enumerate(cashflows))
        npvs.append(npv)
        utils.append(1.0 - np.exp(-a * npv * 1e6))
    npvs = np.array(npvs)
    emv = np.mean(npvs)
    p50 = np.percentile(npvs, 50)
    p10 = np.percentile(npvs, 10)
    p90 = np.percentile(npvs, 90)
    prob_loss = np.mean(npvs < 0.0)
    EU = np.mean(utils)
    CE = -np.log(1.0 - EU) / a / 1e6
    return emv, p50, p10, p90, prob_loss, CE

emv, p50, p10, p90, prob_loss, ce = simulate_project()
print(f"EMV={emv:.1f} P50={p50:.1f} P10={p10:.1f} P90={p90:.1f} "
      f"P(loss)={prob_loss:.2f} CE={ce:.1f}  (millions)")
