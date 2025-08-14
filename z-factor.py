import math
from math import exp
from typing import Tuple
from dataclasses import dataclass

# --------------------------
# Correlations and helpers
# --------------------------

def sutton_pseudocriticals(gamma_g: float) -> Tuple[float, float]:
    """
    Sutton (1985) pseudocritical properties for sweet gas.
    Inputs:
      gamma_g: gas gravity (air = 1).
    Returns:
      Tpc (Rankine), Ppc (psia).
    """
    Tpc = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g**2       # Rankine
    Ppc = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g**2        # psia
    return Tpc, Ppc

def wichert_aziz_correction(Tpc_R: float, Ppc_psia: float,
                            yH2S: float, yCO2: float) -> Tuple[float, float, float]:
    """
    Wichert–Aziz correction for sour gas.
    Inputs:
      Tpc_R, Ppc_psia: Sutton pseudocriticals (sweet).
      yH2S, yCO2: mole fractions.
    Returns:
      (Tpc_corr_R, Ppc_corr_psia, epsilon_R)
    Notes:
      epsilon [Rankine] = 120*(yH2S+yCO2)^0.9 - 15*sqrt(yH2S)
      Tpc' = Tpc - epsilon
      Ppc' = Ppc * Tpc' / (Tpc + yH2S*(1 - yH2S)*epsilon)
      This pressure correction form is a common implementation of Wichert–Aziz.
    """
    eps = 120.0 * (yH2S + yCO2)**0.9 - 15.0 * math.sqrt(max(yH2S, 0.0))
    Tpcp = Tpc_R - eps
    denom = Tpc_R + yH2S * (1.0 - yH2S) * eps
    Ppcp = Ppc_psia * (Tpcp / denom) if denom > 0 else Ppc_psia * (Tpcp / Tpc_R)
    return Tpcp, Ppcp, eps

def F_to_R(TF: float) -> float:
    return TF + 459.67

# --------------------------
# Dranchuk–Abou-Kassem (DAK) Z-correlation
# --------------------------

@dataclass
class DAKCoeffs:
    a1: float = 0.3265
    a2: float = -1.0700
    a3: float = -0.5339
    a4: float = 0.01569
    a5: float = -0.05165
    a6: float = 0.5475
    a7: float = -0.7361
    a8: float = 0.1844
    a9: float = 0.1056
    a10: float = 0.6134
    a11: float = 0.7210

def Z_DAK(Tpr: float, Ppr: float, coeffs: DAKCoeffs = DAKCoeffs()) -> float:
    """
    Compute Z using Dranchuk–Abou-Kassem correlation by solving for reduced density rho_r.
    Equation form (common implementation):
      Ppr = rho*Tpr
            + (A1 + A2/Tpr + A3/Tpr^3 + A4/Tpr^4 + A5/Tpr^5)*rho^2
            + (A6 + A7/Tpr + A8/Tpr^2)*rho^3
            + A9*(1 + A10*rho^2)*rho^2 * exp(-A11*rho^2)
      Then Z = Ppr / (rho * Tpr)
    Robust 1D root solve with safeguarded Newton + bisection on rho in [lo, hi].
    """
    if Tpr <= 0 or Ppr < 0:
        return float('nan')

    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 = (
        coeffs.a1, coeffs.a2, coeffs.a3, coeffs.a4, coeffs.a5,
        coeffs.a6, coeffs.a7, coeffs.a8, coeffs.a9, coeffs.a10, coeffs.a11
    )

    def f(rho: float) -> float:
        term2 = (A1 + A2/Tpr + A3/Tpr**3 + A4/Tpr**4 + A5/Tpr**5) * rho**2
        term3 = (A6 + A7/Tpr + A8/Tpr**2) * rho**3
        term4 = A9 * (1.0 + A10 * rho**2) * rho**2 * math.exp(-A11 * rho**2)
        return rho * Tpr + term2 + term3 + term4 - Ppr

    # Bracket rho_r
    lo, hi = 1e-12, 10.0
    flo, fhi = f(lo), f(hi)
    # If not bracketed, expand hi a bit
    tries = 0
    while flo * fhi > 0 and tries < 10:
        hi *= 2.0
        fhi = f(hi)
        tries += 1
    if flo * fhi > 0:
        # Fall back: ideal-gas Z
        return 1.0

    # Safeguarded Newton–bisection
    rho = 0.5 * (lo + hi)
    for _ in range(80):
        fr = f(rho)
        if abs(fr) < 1e-10:
            break
        # Numerical derivative
        h = max(1e-6, 1e-3 * rho)
        dfr = (f(rho + h) - f(rho - h)) / (2*h)
        # Newton step
        if dfr != 0:
            rho_new = rho - fr / dfr
        else:
            rho_new = rho
        # Keep within bracket; otherwise bisect
        if not (lo < rho_new < hi):
            rho_new = 0.5 * (lo + hi)
        # Update bracket
        if fr > 0:
            hi = rho
        else:
            lo = rho
        rho = rho_new

    if rho <= 0:
        return 1.0
    return Ppr / (rho * Tpr)

# --------------------------
# Main interactive routine
# --------------------------

def main():
    print("Z-factor via Sutton + Wichert–Aziz + DAK")
    try:
        P_psia = float(input("Enter pressure, psia: ").strip())
        T_F    = float(input("Enter temperature, F: ").strip())
        gg     = float(input("Enter gas gravity (air=1): ").strip())
        yH2S   = float(input("Enter mole fraction H2S: ").strip())
        yCO2   = float(input("Enter mole fraction CO2: ").strip())
        yN2    = float(input("Enter mole fraction N2: ").strip())
    except Exception as e:
        print("Invalid input.", e)
        return

    # Basic checks
    if gg <= 0:
        print("Gas gravity must be positive.")
        return
    if any(v < 0 for v in [yH2S, yCO2, yN2]):
        print("Mole fractions must be nonnegative.")
        return
    if yH2S + yCO2 + yN2 > 1.0 + 1e-9:
        print("Sum of H2S+CO2+N2 cannot exceed 1.0.")
        return

    # Pseudocriticals (sweet) from Sutton
    Tpc_R, Ppc_psia = sutton_pseudocriticals(gg)

    # Wichert–Aziz correction for sour gas
    Tpc_corr_R, Ppc_corr_psia, eps_R = wichert_aziz_correction(Tpc_R, Ppc_psia, yH2S, yCO2)

    # Pseudoreduced properties with corrected pseudocriticals
    TR = F_to_R(T_F)
    Tpr = TR / Tpc_corr_R
    Ppr = P_psia / Ppc_corr_psia

    # Z-factor from DAK
    Z = Z_DAK(Tpr, Ppr)

    # Report
    print("\n--- Results ---")
    print(f"Inputs: P = {P_psia:.3f} psia, T = {T_F:.3f} F ({TR:.3f} R), gg = {gg:.5f}, "
          f"yH2S={yH2S:.5f}, yCO2={yCO2:.5f}, yN2={yN2:.5f}")
    print(f"Pseudocriticals (Sutton sweet):    Tpc = {Tpc_R:.3f} R,  Ppc = {Ppc_psia:.3f} psia")
    print(f"Wichert–Aziz epsilon:              eps = {eps_R:.3f} R")
    print(f"Pseudocriticals (sour corrected):  Tpc' = {Tpc_corr_R:.3f} R,  Ppc' = {Ppc_corr_psia:.3f} psia")
    print(f"Pseudoreduced properties:          Tpr = {Tpr:.6f},  Ppr = {Ppr:.6f}")
    print(f"Z-factor (DAK):                    Z   = {Z:.6f}")

if __name__ == "__main__":
    main()