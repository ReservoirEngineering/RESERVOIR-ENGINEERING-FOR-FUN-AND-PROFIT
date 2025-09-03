import math
import matplotlib.pyplot as plt

def sutton_pseudocriticals(gamma_g):
    Tpc_R = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g ** 2
    Ppc_psia = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g ** 2
    return Tpc_R, Ppc_psia

def wichert_aziz_correction(Tpc_R, Ppc_psia, yH2S, yCO2):
    eps_R = 120.0 * (yH2S + yCO2) ** 0.9 - 15.0 * math.sqrt(max(yH2S, 0.0))
    Tpc_corr_R = Tpc_R - eps_R
    denom = Tpc_R + yH2S * (1.0 - yH2S) * eps_R
    if denom <= 0.0:
        Ppc_corr_psia = Ppc_psia * (Tpc_corr_R / Tpc_R)
    else:
        Ppc_corr_psia = Ppc_psia * (Tpc_corr_R / denom)
    return Tpc_corr_R, Ppc_corr_psia, eps_R

def F_to_R(TF):
    return TF + 459.67

def Z_HallYarborough(Tpr, Ppr, tol=1e-10, maxit=80):
    if Tpr <= 0.0 or Ppr < 0.0:
        return float('nan')
    invT = 1.0 / Tpr
    A = 0.06125 * Ppr * math.exp(-1.2 * (1.0 - invT)**2)
    B = invT * (14.76 - 9.76 * invT + 4.58 * invT * invT)
    C = invT * (90.7 - 242.2 * invT + 42.4 * invT * invT)
    D = 2.18 + 2.82 * invT

    def RHS(y):
        if y == 1.0:
            y = 1.0 - 1e-12
        term = (y + y * y + y ** 3 - y ** 4) / ((1.0 - y) ** 3)
        return term - B * y * y + C * (y ** D)

    # Newton-Raphson Safeguarded
    y = max(min(A, 2.0), 1e-10)
    if abs(y - 1.0) < 1e-6:
        y = 0.999999

    lo, hi = 1e-12, 5.0
    for it in range(maxit):
        fy = RHS(y) - A
        if abs(fy) < tol:
            break
        h = max(1e-8, 1e-3 * y)
        yp, ym = y + h, max(y - h, 1e-12)
        if abs(yp - 1.0) < 1e-12: yp = 1.0 + 1e-12
        if abs(ym - 1.0) < 1e-12: ym = 1.0 - 1e-12
        df = (RHS(yp) - RHS(ym)) / (yp - ym)
        if df == 0.0 or not math.isfinite(df):
            y_new = 0.5 * (lo + hi)
        else:
            y_new = y - fy / df
        if not (lo < y_new < hi) or abs(y_new - 1.0) < 1e-9:
            if fy > 0:
                hi = min(hi, y)
            else:
                lo = max(lo, y)
            y_new = 0.5 * (lo + hi)
        y = y_new

    if not math.isfinite(y) or y <= 0.0 or abs(RHS(y) - A) > 1e-6:
        grid1 = [1e-10 + i * (0.98 - 1e-10) / 200 for i in range(201)]
        grid2 = [1.02 + i * (3.0 - 1.02) / 200 for i in range(201)]
        grid = grid1 + grid2
        vals = [RHS(yg) - A for yg in grid]
        br_lo, br_hi = None, None
        for i in range(len(grid) - 1):
            if vals[i] == 0.0:
                y = grid[i]
                break
            if vals[i] * vals[i + 1] < 0.0:
                br_lo, br_hi = grid[i], grid[i + 1]
                break
        if br_lo is not None:
            for _ in range(60):
                mid = 0.5 * (br_lo + br_hi)
                fm = RHS(mid) - A
                if abs(fm) < tol:
                    y = mid
                    break
                if fm * (RHS(br_lo) - A) > 0.0:
                    br_lo = mid
                else:
                    br_hi = mid
            y = 0.5 * (br_lo + br_hi)
        if y <= 0.0 or not math.isfinite(y):
            return 1.0
    Z = A / y
    return Z
def main():
    print("Z-factor via Sutton + Wichert–Aziz + Hall–Yarborough")
    try:
        Pmax_psia = float(input("Enter MAXIMUM pressure, psia: ").strip())
        T_F = float(input("Enter temperature, F: ").strip())
        gg = float(input("Enter gas gravity (air=1): ").strip())
        yH2S = float(input("Enter mole fraction H2S: ").strip())
        yCO2 = float(input("Enter mole fraction CO2: ").strip())
        yN2 = float(input("Enter mole fraction N2: ").strip())
    except Exception as e:
        print("Invalid input.", e)
        return
    if gg <= 0 or yH2S < 0 or yCO2 < 0 or yN2 < 0 or (yH2S + yCO2 + yN2) > 1.0 + 1e-9:
        print("Check inputs: gg>0; mole fractions nonnegative; yH2S+yCO2+yN2 <= 1.0.")
        return

    Tpc_R, Ppc_psia = sutton_pseudocriticals(gg)
    Tpc_corr_R, Ppc_corr_psia, eps_R = wichert_aziz_correction(Tpc_R, Ppc_psia, yH2S, yCO2)
    TR = F_to_R(T_F)
    Tpr = TR / Tpc_corr_R

    # Pressure range: start at 100 psia
    n_points = 20
    Pmin_psia = 100.0
    if Pmax_psia < Pmin_psia:
        print("Max pressure must be at least 100 psia.")
        return
    Ps = [Pmin_psia + (Pmax_psia - Pmin_psia) * i / (n_points - 1) for i in range(n_points)]
    Zs = []

    for P_psia in Ps:
        Ppr = P_psia / Ppc_corr_psia
        Z = Z_HallYarborough(Tpr, Ppr)
        Zs.append(Z)

    print("\n--- Results (Hall–Yarborough) ---")
    print(f"Inputs: [MaxPressure] = {Pmax_psia:.3f} psia, T = {T_F:.3f} F ({TR:.3f} R), gg = {gg:.5f}, "
          f"yH2S={yH2S:.5f}, yCO2={yCO2:.5f}, yN2={yN2:.5f}")
    print(f"Pseudocriticals (Sutton sweet): Tpc = {Tpc_R:.3f} R, Ppc = {Ppc_psia:.3f} psia")
    print(f"Wichert–Aziz epsilon: eps = {eps_R:.3f} R")
    print(f"Pseudocriticals (sour corrected): Tpc' = {Tpc_corr_R:.3f} R, Ppc' = {Ppc_corr_psia:.3f} psia")
    print(f"Pseudoreduced properties: Tpr = {Tpr:.6f}")
    print(f"Plotting Z-factor vs P...")

    with open("HYout.dat", "w") as f:
        f.write("Hall-Yarborough Z-Factor Output\n")
        f.write("Input Summary:\n")
        f.write(f"  MaxPressure (psia): {Pmax_psia:.3f}\n")
        f.write(f"  Temperature (F): {T_F:.3f} ({TR:.3f} R)\n")
        f.write(f"  Gas Gravity: {gg:.5f}\n")
        f.write(f"  yH2S: {yH2S:.5f}   yCO2: {yCO2:.5f}   yN2: {yN2:.5f}\n")
        f.write("\nPseudocriticals (Sutton sweet):\n")
        f.write(f"  Tpc = {Tpc_R:.3f} R, Ppc = {Ppc_psia:.3f} psia\n")
        f.write(f"Wichert–Aziz correction:\n")
        f.write(f"  epsilon = {eps_R:.3f} R\n")
        f.write(f"Pseudocriticals (sour corrected):\n")
        f.write(f"  Tpc' = {Tpc_corr_R:.3f} R, Ppc' = {Ppc_corr_psia:.3f} psia\n")
        f.write(f"Pseudoreduced temperature: Tpr = {Tpr:.6f}\n\n")
        f.write("Pressure (psia)\tZ-Factor\n")
        for P, Z in zip(Ps, Zs):
            f.write(f"{P:.3f}\t{Z:.6f}\n")

    plt.figure()
    plt.plot(Ps, Zs, marker='o')
    plt.title("Hall-Yarborough Z-factor vs Pressure")
    plt.xlabel("Pressure (psia)")
    plt.ylabel("Z-Factor")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("HYout.png")
    plt.show()
    print("Results written to HYout.dat and plot saved as HYout.png.")


if __name__ == "__main__":
    main()
