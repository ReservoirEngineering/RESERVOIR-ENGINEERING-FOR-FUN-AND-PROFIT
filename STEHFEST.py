import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.special import k0, k1, i0, i1, i0e, i1e, k0e, k1e, expi
from matplotlib.ticker import LogLocator, NullFormatter

# -------------------------------
# Stehfest inversion (robust weights)
# f(t) ≈ (ln 2)/t * sum_{k=1}^{N} V_k * F(k ln 2 / t),  N even
# -------------------------------
def stehfest_weights(N):
    if N % 2 != 0:
        raise ValueError("N must be even.")
    V = np.zeros(N, dtype=float)
    n2 = N // 2
    for k in range(1, N + 1):
        ssum = 0.0
        jmin = (k + 1) // 2
        jmax = min(k, n2)
        for j in range(jmin, jmax + 1):
            # V_k = (-1)^{n2+k} sum_j [ j^{n2} (2j)! / ((n2-j)! j! (j-1)! (k-j)! (2j-k)!) ]
            num = (j**n2) * factorial(2*j)
            den = factorial(n2 - j) * factorial(j) * factorial(j - 1) * factorial(k - j) * factorial(2*j - k)
            ssum += num / den
        V[k - 1] = ((-1)**(n2 + k)) * ssum
    return V

def stehfest_invert(F, t, N=12):
    ln2 = np.log(2.0)
    V = stehfest_weights(N)
    acc = 0.0
    for k in range(1, N + 1):
        s = k * ln2 / t
        acc += V[k - 1] * F(s)
    return (ln2 / t) * acc

def invert_over_times(F, t_array, N=12):
    out = np.empty_like(t_array, dtype=float)
    for i, t in enumerate(t_array):
        out[i] = stehfest_invert(F, t, N=N)
    return out

# -------------------------------
# Infinite-acting line-source Laplace kernel:
#   \bar{p}_D(s) = 2 K0(sqrt(s)) / ( s * sqrt(s) * K1(sqrt(s)) )
# -------------------------------
def pDbar_infinite(s):
    z = np.sqrt(s)
    return 2.0 * k0(z) / (s * z * k1(z))

# Analytic infinite-acting line-source (time domain) for validation:
#   p_D(t_D) = -Ei(-1/(4 t_D))
def pD_infinite_analytic(td):
    return -expi(-1.0 / (4.0 * td))

# -------------------------------
# No-flow outer boundary at r_eD:
# General Laplace-domain solution for well response:
#   \bar{p}_D(s; r_eD) =
#     2 [ K0(z) I1(z_e) + I0(z) K1(z_e) ] / { s z [ I1(z_e) K1(z) - K1(z_e) I1(z) ] }
# where z = sqrt(s), z_e = z * r_eD.
# To avoid overflow for large z_e, use scaled modified Bessels:
#   Iν(x) = iνe(x) * e^{x},  Kν(x) = kνe(x) * e^{-x}.
# After factoring, the exponentials cancel analytically; numerically we use:
#   Numerator ~ 2 [ k0(z) * i1e(z_e) + I0(z) * k1e(z_e) * e^{-2 z_e} ]
#   Denominator ~ s z [ i1e(z_e) * K1(z) - k1e(z_e) * e^{-2 z_e} * I1(z) ]
# -------------------------------
def pDbar_noflow(s, r_eD):
    z = np.sqrt(s)
    ze = z * r_eD
    # Scaled terms at large ze
    i1e_ze = i1e(ze)
    k1e_ze = k1e(ze)
    e_m2ze = np.exp(-2.0 * ze)
    # Unscaled at small z are OK
    num = 2.0 * (k0(z) * i1e_ze + i0(z) * k1e_ze * e_m2ze)
    den = s * z * (i1e_ze * k1(z) - k1e_ze * e_m2ze * i1(z))
    return num / den

# -------------------------------
# Build dataset across decades
# -------------------------------
td = np.logspace(-2, 9, 600)  # 1e-2 to 1e9

# Infinite-acting (Stehfest and analytic for validation)
Pd_inf = invert_over_times(pDbar_infinite, td, N=12)
Pd_inf_an = pD_infinite_analytic(td)

# No-flow outer boundaries
re_list = [10.0, 100.0, 1000.0, 10000.0]
Pd_nf = {}
for reD in re_list:
    Pd_nf[reD] = invert_over_times(lambda s: pDbar_noflow(s, reD), td, N=12)

# -------------------------------
# Save combined dataset to CSV
# Columns: td, Pd_infinite, Pd_infinite_analytic, Pd_reD10, Pd_reD100, Pd_reD1000, Pd_reD10000
# -------------------------------
cols = [td, Pd_inf, Pd_inf_an] + [Pd_nf[reD] for reD in re_list]
data = np.column_stack(cols)
header_cols = ["td", "Pd_infinite", "Pd_infinite_analytic"] + [f"Pd_reD{int(reD)}" for reD in re_list]
np.savetxt("pd_vs_td_bounded.csv", data, delimiter=",", header=",".join(header_cols), comments="")
print("Wrote pd_vs_td_bounded.csv")

# Print first 8 rows for quick inspection
print("\nHead of table:")
for i in range(8):
    row = [f"{td[i]:.6g}", f"{Pd_inf[i]:.6g}", f"{Pd_inf_an[i]:.6g}"] + [f"{Pd_nf[reD][i]:.6g}" for reD in re_list]
    print(", ".join(row))

# -------------------------------
# Plot: log-log with equal log-cycle lengths, y-limits 1–100
# -------------------------------
x_min, x_max = 1e-2, 1e9
y_min, y_max = 1.0, 100.0

# Masks for finite positive values
mask_inf = np.isfinite(Pd_inf) & (Pd_inf > 0)
masks_nf = {reD: (np.isfinite(Pd_nf[reD]) & (Pd_nf[reD] > 0)) for reD in re_list}
mask_ana = np.isfinite(Pd_inf_an) & (Pd_inf_an > 0)

fig, ax = plt.subplots(figsize=(7, 4))

# Analytic infinite-acting (reference)
ax.loglog(td[mask_ana], Pd_inf_an[mask_ana], 'k-', lw=2.0, label='Infinite (analytic)')

# Stehfest infinite-acting
ax.loglog(td[mask_inf], Pd_inf[mask_inf], color='tab:red', ls='--', lw=1.3, alpha=0.9, label='Infinite (Stehfest)')

# No-flow outer boundary curves
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
for color, reD in zip(colors, re_list):
    m = masks_nf[reD]
    ax.loglog(td[m], Pd_nf[reD][m], color=color, lw=1.6, label=fr'No-flow $r_{{eD}}={int(reD)}$')

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel(r"$t_D$")
ax.set_ylabel(r"$p_D$")
ax.grid(True, which="both", ls=":", alpha=0.6)
ax.legend(loc="best", fontsize=9)

# Make equal log cycles on x and y (square decades)
nx = np.log10(x_max) - np.log10(x_min)
ny = np.log10(y_max) - np.log10(y_min)
ax.set_box_aspect(ny / nx)  # Matplotlib >= 3.3

# Log ticks
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

plt.tight_layout()
plt.show()
