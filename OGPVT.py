import math
import numpy as np
import matplotlib.pyplot as plt

# Inputs
Pi = 3000.0           # psia
Pb_in = 2422.394208   # psia (provided)
Pab = 200.0           # psia
gamma_g = 0.66        # gas specific gravity (air=1)
T_F = 155.0           # F
API = 35.5            # deg API
phi = 0.125           # -
h = 35.0              # ft
A_acres = 1765.0      # acres
Swi = 0.32            # -
Rsi = 561.0           # scf/STB
c_oa = 1.5e-5         # oil compressibility above Pb (psi^-1), per request

# Derived
gamma_o = 141.5 / (API + 131.5)
T_R = T_F + 459.67

# Standing correlations
def standing_pb_from_rsi(API, gamma_g, T_F, Rsi):
    A = 0.0125*API - 0.00091*T_F
    term = (Rsi/gamma_g)**(1.0/1.2048) / (10.0**A)
    Pb = 18.2 * (term - 1.4)
    return Pb

def standing_rs(P, API, gamma_g, T_F):
    A = 0.0125*API - 0.00091*T_F
    return gamma_g * ((P/18.2 + 1.4) * (10.0**A))**1.2048

def standing_bo_saturated(Rs, T_F, gamma_g, gamma_o):
    term = Rs * math.sqrt(gamma_g/gamma_o) + 1.25*T_F
    return 0.9759 + 0.00012 * (term**1.2)

# Pseudocritical properties (Sutton)
def sutton_pseudocriticals(gamma_g):
    Tpc = 169.2 + 349.5*gamma_g - 74.0*(gamma_g**2)   # Rankine
    Ppc = 756.8 - 131.0*gamma_g - 3.6*(gamma_g**2)    # psia
    return Tpc, Ppc

# Dranchuk–Abou-Kassem Z-factor (iterative on reduced density)
def z_dak(P, T_R, gamma_g):
    Tpc, Ppc = sutton_pseudocriticals(gamma_g)
    Ppr = P / Ppc
    Tpr = T_R / Tpc
    if Ppr <= 0.0:
        return 1.0

    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210

    def Ppr_from_rho(rho_r, Tpr):
        A = A1 + A2/Tpr + A3/(Tpr**3) + A4/(Tpr**4) + A5/(Tpr**5)
        B = A6 + A7/Tpr + A8/(Tpr**2)
        exp_term = math.exp(-A11 * (rho_r**2))
        termE = (1.0 + A10*(rho_r**2)) * (rho_r**2) * exp_term
        Ppr_est = rho_r*Tpr + A*(rho_r**2) + B*(rho_r**3) + A9*termE
        dPpr_drho = Tpr + 2.0*A*rho_r + 3.0*B*(rho_r**2) + A9 * exp_term * (2.0*rho_r + 4.0*A10*(rho_r**3) - 2.0*A11*rho_r*((rho_r**2) + A10*(rho_r**4)))
        return Ppr_est, dPpr_drho

    rho_r = max(0.27 * Ppr / Tpr, 1e-8)
    for _ in range(60):
        Ppr_est, dPpr_drho = Ppr_from_rho(rho_r, Tpr)
        F = Ppr_est - Ppr
        if abs(F) < 1e-10:
            break
        if dPpr_drho == 0.0 or not math.isfinite(dPpr_drho):
            dPpr_drho = 1e-6
        step = F / dPpr_drho
        rho_new = rho_r - step
        if rho_new <= 0.0 or not math.isfinite(rho_new):
            rho_new = max(rho_r * 0.5, 1e-8)
        # mild damping to enhance robustness
        if abs(rho_new - rho_r) > 0.5:
            rho_new = rho_r + 0.5 * math.copysign(1.0, rho_new - rho_r)
        rho_r = rho_new

    Z = Ppr / (rho_r * Tpr)
    if not math.isfinite(Z) or Z <= 0.0:
        Z = 1.0
    return Z

def bg_from_z(Z, P, T_R):
    return 0.0282793 * Z * T_R / P  # rb/scf

# Gas viscosity (Lee-Gonzalez-Eakin), density in g/cc
def gas_density_lbmft3(P, T_R, Z, gamma_g):
    return (28.97 * gamma_g * P) / (Z * 10.73 * T_R)

def gas_density_gcc(P, T_R, Z, gamma_g):
    rho_lbmft3 = gas_density_lbmft3(P, T_R, Z, gamma_g)
    return rho_lbmft3 / 62.4279606

def mu_g_lee(P, T_R, Z, gamma_g):
    M = 28.97 * gamma_g
    rho_g = gas_density_gcc(P, T_R, Z, gamma_g)
    K = ((9.379 + 0.01607*M) * (T_R**1.5)) / (209.2 + 19.26*M + T_R)
    X = 3.448 + (986.4 / T_R) + 0.01009*M
    Y = 2.447 - 0.2224*X
    expo = X * (rho_g**Y)
    expo = min(expo, 50.0)
    mu = 1.0e-4 * K * math.exp(expo)
    return mu  # cp

# Simple live-oil viscosity placeholder (can replace with Beggs/Robinson later)
def mu_o_model(P, Pb, mu_ob=0.7):
    if P >= Pb:
        return mu_ob * (1.0 + 0.05*(P - Pb)/Pb)
    else:
        n = 0.6
        return mu_ob * ((Pb / max(P, 1.0))**n)

# Build PVT tables vs pressure
Pmin = Pab
Pmax = Pi
npts = 101
P_arr = np.linspace(Pmin, Pmax, npts)

Pb_calc = standing_pb_from_rsi(API, gamma_g, T_F, Rsi)
Pb = Pb_calc
Rsb = Rsi

Rs_arr = np.zeros_like(P_arr)
Bo_arr = np.zeros_like(P_arr)
muo_arr = np.zeros_like(P_arr)
Z_arr = np.zeros_like(P_arr)
Bg_arr = np.zeros_like(P_arr)
mug_arr = np.zeros_like(P_arr)

Bo_b = standing_bo_saturated(Rsb, T_F, gamma_g, gamma_o)

for i, P in enumerate(P_arr):
    if P <= Pb:
        Rs = standing_rs(P, API, gamma_g, T_F)
    else:
        Rs = Rsb
    Rs_arr[i] = Rs

    if P <= Pb:
        Bo_arr[i] = standing_bo_saturated(Rs, T_F, gamma_g, gamma_o)
    else:
        Bo_arr[i] = Bo_b * math.exp(-c_oa * (P - Pb))

    muo_arr[i] = mu_o_model(P, Pb, mu_ob=0.7)

    Z = z_dak(P, T_R, gamma_g)
    Z_arr[i] = Z
    Bg_arr[i] = bg_from_z(Z, P, T_R)
    mug_arr[i] = mu_g_lee(P, T_R, Z, gamma_g)

# OOIP using Bo at initial pressure
idx_pi = int(np.argmin(np.abs(P_arr - Pi)))
Boi = Bo_arr[idx_pi]

bulk_acreft = A_acres * h
bulk_bbl = bulk_acreft * 7758.0
pore_bbl = bulk_bbl * phi
oil_res_bbl = pore_bbl * (1.0 - Swi)
OOIP_STB = oil_res_bbl / Boi

# Report
print(f"Oil specific gravity (gamma_o): {gamma_o:.4f}")
print(f"Temperature (R): {T_R:.2f}")
print(f"Bubblepoint (Standing calc): {Pb_calc:.1f} psia; provided: {Pb_in:.1f} psia")
print(f"Bo at bubblepoint: {Bo_b:.3f} RB/STB")
print(f"Bo at Pi={Pi:.0f} psia: {Boi:.3f} RB/STB (c_o={c_oa:.2e} 1/psi above Pb)")
print(f"OOIP: {OOIP_STB/1e6:.2f} MMSTB")

# Plots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.ravel()

axs[0].plot(P_arr, Rs_arr)
axs[0].set_xlabel("Pressure (psia)")
axs[0].set_ylabel("Rs (scf/STB)")
axs[0].set_title("Solution GOR Rs vs P")

axs[1].plot(P_arr, Bo_arr)
axs[1].set_xlabel("Pressure (psia)")
axs[1].set_ylabel("Bo (RB/STB)")
axs[1].set_title("Oil FVF Bo vs P")

axs[2].plot(P_arr, muo_arr)
axs[2].set_xlabel("Pressure (psia)")
axs[2].set_ylabel("Oil viscosity (cp)")
axs[2].set_title("Oil viscosity vs P")

axs[3].plot(P_arr, Z_arr)
axs[3].set_xlabel("Pressure (psia)")
axs[3].set_ylabel("Z-factor (-)")
axs[3].set_title("Gas Z-factor vs P (DAK)")

axs[4].plot(P_arr, Bg_arr)
axs[4].set_xlabel("Pressure (psia)")
axs[4].set_ylabel("Bg (rb/scf)")
axs[4].set_title("Gas FVF Bg vs P")

axs[5].plot(P_arr, mug_arr)
axs[5].set_xlabel("Pressure (psia)")
axs[5].set_ylabel("Gas viscosity (cp)")
axs[5].set_title("Gas viscosity vs P (Lee et al.)")

fig.tight_layout()
fig.savefig("pvt_properties.png", dpi=160)
plt.show()
