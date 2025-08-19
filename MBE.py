import math
import numpy as np
import matplotlib.pyplot as plt

# Inputs
Pi = 3000.0
Pb_in = 2422.394208
Pmin = 400.0
dP_step = 50.0

gamma_g = 0.66
T_F = 155.0
API = 35.5

phi = 0.125
h = 35.0
A_acres = 1765.0
Swi = 0.32

Rsi = 561.0
c_oa = 1.5e-5  # oil compressibility above Pb (1/psi)

# Relperm params
Sgc = 0.05
Sorg = 0.05
n_og = 2.5
n_g = 2.5
KROMAX = 0.8
KRGMAX = 0.6

# Derived
gamma_o = 141.5 / (API + 131.5)
T_R = T_F + 459.67
acreft_to_rb = 7758.0
bulk_acreft = A_acres * h
PV_rb = bulk_acreft * acreft_to_rb * phi
Vw_rb = PV_rb * Swi

# PVT correlations
def standing_pb_from_rsi(API, gamma_g, T_F, Rsi):
    A = 0.0125*API - 0.00091*T_F
    term = (Rsi/gamma_g)**(1.0/1.2048) / (10.0**A)
    return 18.2 * (term - 1.4)

def standing_rs(P, API, gamma_g, T_F):
    A = 0.0125*API - 0.00091*T_F
    return gamma_g * ((P/18.2 + 1.4) * (10.0**A))**1.2048

def standing_bo_saturated(Rs, T_F, gamma_g, gamma_o):
    term = Rs * math.sqrt(gamma_g/gamma_o) + 1.25*T_F
    return 0.9759 + 0.00012 * (term**1.2)

def sutton_pseudocriticals(gamma_g):
    Tpc = 169.2 + 349.5*gamma_g - 74.0*(gamma_g**2)
    Ppc = 756.8 - 131.0*gamma_g - 3.6*(gamma_g**2)
    return Tpc, Ppc

# Dranchuk–Abou-Kassem (DAK) Z-factor
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
        if abs(F) < 1e-12:
            break
        if not math.isfinite(dPpr_drho) or dPpr_drho == 0.0:
            dPpr_drho = 1e-6
        step = F / dPpr_drho
        rho_new = rho_r - step
        if not math.isfinite(rho_new) or rho_new <= 0.0:
            rho_new = max(rho_r*0.5, 1e-8)
        if abs(rho_new - rho_r) > 0.5:
            rho_new = rho_r + 0.5 * math.copysign(1.0, rho_new - rho_r)
        rho_r = rho_new

    Z = Ppr / (rho_r * Tpr)
    if not math.isfinite(Z) or Z <= 0.0:
        Z = 1.0
    return Z

def bg_from_z(Z, P, T_R):
    return 0.0282793 * Z * T_R / P

def gas_density_gcc(P, T_R, Z, gamma_g):
    rho_lbmft3 = (28.97 * gamma_g * P) / (Z * 10.73 * T_R)
    return rho_lbmft3 / 62.4279606

def mu_g_lee(P, T_R, Z, gamma_g):
    M = 28.97 * gamma_g
    rho_g = gas_density_gcc(P, T_R, Z, gamma_g)
    K = ((9.379 + 0.01607*M) * (T_R**1.5)) / (209.2 + 19.26*M + T_R)
    X = 3.448 + (986.4 / T_R) + 0.01009*M
    Y = 2.447 - 0.2224*X
    expo = X * (rho_g**Y)
    if expo > 50.0:
        expo = 50.0
    return 1.0e-4 * K * math.exp(expo)

# Oil viscosity (placeholder for now; will swap to Beggs & Robinson on your confirmation)
def mu_o_model(P, Pb, mu_ob=0.7):
    if P >= Pb:
        return mu_ob * (1.0 + 0.05*(P - Pb)/Pb)
    else:
        n = 0.6
        return mu_ob * ((Pb / max(P, 1.0))**n)

def pvt_at_P(P, Pb, Rsb, Bbo, c_oa):
    if P >= Pb:
        Rs = Rsb
        Bo = Bbo * math.exp(-c_oa * (P - Pb))
    else:
        Rs = standing_rs(P, API, gamma_g, T_F)
        Bo = standing_bo_saturated(Rs, T_F, gamma_g, gamma_o)
    Z = z_dak(P, T_R, gamma_g)
    Bg = bg_from_z(Z, P, T_R)
    muo = mu_o_model(P, Pb)
    mug = mu_g_lee(P, T_R, Z, gamma_g)
    return Rs, Bo, Bg, muo, mug

def relperm_go(Sw, Sg, Sgc, Sorg, n_og, n_g, KROMAX, KRGMAX):
    So = max(0.0, 1.0 - Sw - Sg)
    denom = max(1e-12, 1.0 - Sw - Sorg - Sgc)
    Sg_eff = (Sg - Sgc) / denom
    So_eff = (So - Sorg) / denom
    if Sg_eff < 0.0:
        Sg_eff = 0.0
    if Sg_eff > 1.0:
        Sg_eff = 1.0
    if So_eff < 0.0:
        So_eff = 0.0
    if So_eff > 1.0:
        So_eff = 1.0
    k_rg = KRGMAX * (Sg_eff**n_g) if Sg_eff > 0.0 else 0.0
    k_ro = KROMAX * (So_eff**n_og) if So_eff > 0.0 else 0.0
    return k_ro, k_rg

def pressure_schedule(Pi, Pb, Pmin, dP):
    p_up = []
    P = Pi
    while P - dP > Pb:
        p_up.append(P)
        P -= dP
    p_up.append(P)
    if p_up[-1] != Pb:
        p_up.append(Pb)

    p_dn = []
    P = Pb - dP
    while P > Pmin:
        p_dn.append(P)
        P -= dP
    if len(p_dn) == 0 or p_dn[-1] != Pmin:
        p_dn.append(Pmin)
    return p_up, p_dn

# Precompute Pb, Bo_b, Boi, OOIP
Pb = standing_pb_from_rsi(API, gamma_g, T_F, Rsi)
Rsb = Rsi
Bo_b = standing_bo_saturated(Rsb, T_F, gamma_g, gamma_o)
Rs_i, Boi, _, _, _ = pvt_at_P(Pi, Pb, Rsb, Bo_b, c_oa)

N_STB = (PV_rb - Vw_rb) / Boi
C_pv = PV_rb - Vw_rb  # constant PV constraint term

def step_above_pb(P1, P2, Np1, Gp1):
    Rs1, Bo1, _, _, _ = pvt_at_P(P1, Pb, Rsb, Bo_b, c_oa)
    Rs2, Bo2, _, _, _ = pvt_at_P(P2, Pb, Rsb, Bo_b, c_oa)
    Np2 = N_STB - (N_STB - Np1) * (Bo1 / Bo2)
    if Np2 < Np1:
        Np2 = Np1
    dNp = Np2 - Np1
    dGp = Rsi * dNp
    Gp2 = Gp1 + dGp
    return dNp, dGp, Np2, Gp2

def step_below_pb(P1, P2, Np1, Gp1):
    Rs1, Bo1, Bg1, muo1, mug1 = pvt_at_P(P1, Pb, Rsb, Bo_b, c_oa)
    Rs2, Bo2, Bg2, muo2, mug2 = pvt_at_P(P2, Pb, Rsb, Bo_b, c_oa)

    Rs_avg = 0.5 * (Rs1 + Rs2)
    Bo_avg = 0.5 * (Bo1 + Bo2)
    Bg_avg = 0.5 * (Bg1 + Bg2)
    muo_avg = 0.5 * (muo1 + muo2)
    mug_avg = 0.5 * (mug1 + mug2)

    Rp_inc = Rs_avg

    for _ in range(20):
        A11 = -Bo2
        A12 = Bg2
        b1 = C_pv - N_STB * Bo2

        A21 = (Rp_inc - Rs2)
        A22 = 1.0
        b2 = N_STB * (Rsi - Rs2) - (Gp1 - Rp_inc * Np1)

        D = A11 * A22 - A12 * A21
        if abs(D) < 1e-14:
            D = -1e-14

        Np2 = (b1 * A22 - A12 * b2) / D
        Gf2 = (A11 * b2 - A21 * b1) / D

        if Np2 < Np1:
            Np2 = Np1
        if Gf2 < 0.0:
            Gf2 = 0.0

        dNp = Np2 - Np1

        Vg2_rb = C_pv - (N_STB - Np2) * Bo2
        if Vg2_rb < 0.0:
            Vg2_rb = 0.0
        Sg_end = Vg2_rb / PV_rb

        if Sg_end <= Sgc or dNp <= 0.0:
            Rp_new = Rs_avg
        else:
            kro, krg = relperm_go(Swi, Sg_end, Sgc, Sorg, n_og, n_g, KROMAX, KRGMAX)
            lam_o = kro / max(muo_avg, 1e-12)
            lam_g = krg / max(mug_avg, 1e-12)
            if (lam_g + lam_o) > 0.0:
                fg = lam_g / (lam_g + lam_o)
            else:
                fg = 0.0
            if fg > 0.98:
                fg = 0.98
            Rp_ff = Rs_avg + (Bo_avg / Bg_avg) * (fg / (1.0 - fg))
            Rp_new = 0.5 * Rp_inc + 0.5 * Rp_ff

        if abs(Rp_new - Rp_inc) < 1.0:
            Rp_inc = Rp_new
            break

        Rp_inc = Rp_new

    dNp = Np2 - Np1
    dGp = Rp_inc * dNp
    Gp2 = Gp1 + dGp
    return dNp, dGp, Np2, Gp2

# Pressure schedule
p_up, p_dn = pressure_schedule(Pi, Pb, Pmin, dP_step)

# Forecast arrays
P_list = []
dNp_list = []
dGp_list = []
Np_list = []
Gp_list = []
GOR_prod_list = []

# Initialize
Np_cum = 0.0
Gp_cum = 0.0

# Above Pb
for i in range(1, len(p_up)):
    P1 = p_up[i-1]
    P2 = p_up[i]
    dNp, dGp, Np_cum, Gp_cum = step_above_pb(P1, P2, Np_cum, Gp_cum)
    P_list.append(P2)
    dNp_list.append(dNp)
    dGp_list.append(dGp)
    Np_list.append(Np_cum)
    Gp_list.append(Gp_cum)
    gor_prod = dGp / dNp if dNp > 0 else Rsi
    GOR_prod_list.append(gor_prod)

# Below Pb
if len(p_dn) > 0:
    P_prev = p_up[-1]
    for P2 in p_dn:
        dNp, dGp, Np_cum, Gp_cum = step_below_pb(P_prev, P2, Np_cum, Gp_cum)
        P_list.append(P2)
        dNp_list.append(dNp)
        dGp_list.append(dGp)
        Np_list.append(Np_cum)
        Gp_list.append(Gp_cum)
        gor_prod = dGp / dNp if dNp > 0 else standing_rs(P2, API, gamma_g, T_F)
        GOR_prod_list.append(gor_prod)
        P_prev = P2

# Output
print("Pressure(psia), dNp(STB), dGp(scf), Np_cum(STB), Gp_cum(scf), Producing GOR (scf/STB)")
for i in range(len(P_list)):
    gor = dGp_list[i] / dNp_list[i] if dNp_list[i] > 0 else Rsi
    print(f"{P_list[i]:8.2f}, {dNp_list[i]:12.2f}, {dGp_list[i]:12.2f}, {Np_list[i]:12.2f}, {Gp_list[i]:12.2f}, {gor:10.2f}")

# Plots
P_arr = np.array(P_list)
Np_arr = np.array(Np_list)
Gp_arr = np.array(Gp_list)
GOR_arr = np.array(GOR_prod_list)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

axs[0].plot(P_arr, Np_arr/1e6, marker='o')
axs[0].invert_xaxis()
axs[0].set_xlabel("Pressure (psia)")
axs[0].set_ylabel("Np (MMSTB)")
axs[0].set_title("Cumulative oil vs Pressure")

axs[1].plot(P_arr, Gp_arr/1e9, marker='o', color='tab:orange')
axs[1].invert_xaxis()
axs[1].set_xlabel("Pressure (psia)")
axs[1].set_ylabel("Gp (Bscf)")
axs[1].set_title("Cumulative gas vs Pressure")

axs[2].plot(P_arr, GOR_arr, marker='o', color='tab:green')
axs[2].invert_xaxis()
axs[2].set_xlabel("Pressure (psia)")
axs[2].set_ylabel("Producing GOR (scf/STB)")
axs[2].set_title("Producing GOR vs Pressure")

fig.tight_layout()
fig.savefig("mbal_forecast.png", dpi=160)
plt.show()
