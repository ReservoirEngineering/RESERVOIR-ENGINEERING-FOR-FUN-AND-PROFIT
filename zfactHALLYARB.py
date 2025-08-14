import math

def sutton_pseudocriticals(gamma_g):
# Sutton (1985), sweet gas
Tpc_R = 169.2 + 349.5gamma_g - 74.0gamma_g2
Ppc_psia = 756.8 - 131.0gamma_g - 3.6gamma_g2
return Tpc_R, Ppc_psia

def wichert_aziz_correction(Tpc_R, Ppc_psia, yH2S, yCO2):
# Wichert–Aziz correction
eps_R = 120.0 * (yH2S + yCO2)**0.9 - 15.0 * math.sqrt(max(yH2S, 0.0))
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
"""
Hall–Yarborough (1973) Z-factor.
Solve for reduced density y from: A = RHS(y)
A = 0.06125 * Ppr * exp(-1.2 * (1 - 1/Tpr)^2)
B = (1/Tpr) * (14.76 - 9.76*(1/Tpr) + 4.58*(1/Tpr)^2)
C = (1/Tpr) * (90.7 - 242.2*(1/Tpr) + 42.4*(1/Tpr)^2)
D = 2.18 + 2.82*(1/Tpr)
RHS(y) = (y + y^2 + y^3 - y^4)/(1 - y)^3 - By^2 + Cy**D
Then Z = A / y.
"""
if Tpr <= 0.0 or Ppr < 0.0:
return float('nan')

invT = 1.0 / Tpr
A = 0.06125 * Ppr * math.exp(-1.2 * (1.0 - invT)**2)
B = invT * (14.76 - 9.76*invT + 4.58*invT*invT)
C = invT * (90.7 - 242.2*invT + 42.4*invT*invT)
D = 2.18 + 2.82*invT

def RHS(y):
if y == 1.0:
y = 1.0 - 1e-12
term = (y + y*y + y**3 - y**4) / ((1.0 - y)**3)
return term - B*y*y + C*(y**D)

# Solve A = RHS(y) for y > 0
# Initial guess near ideal: y0 ≈ A
y = max(min(A, 2.0), 1e-10)
if abs(y - 1.0) < 1e-6:
y = 0.999999

# Safeguarded Newton with simple bounds
lo, hi = 1e-12, 5.0 # y typically <~ a few
for it in range(maxit):
fy = RHS(y) - A
if abs(fy) < tol:
break
# numeric derivative
h = max(1e-8, 1e-3*y)
yp, ym = y + h, max(y - h, 1e-12)
# avoid y=1 in derivative eval
if abs(yp - 1.0) < 1e-12: yp = 1.0 + 1e-12
if abs(ym - 1.0) < 1e-12: ym = 1.0 - 1e-12
df = (RHS(yp) - RHS(ym)) / (yp - ym)
if df == 0.0 or not math.isfinite(df):
y_new = 0.5*(lo + hi)
else:
y_new = y - fy/df
# keep inside bounds and away from singularity at y=1
if not (lo < y_new < hi) or abs(y_new - 1.0) < 1e-9:
# adjust bracket based on sign
if fy > 0:
hi = min(hi, y)
else:
lo = max(lo, y)
y_new = 0.5*(lo + hi)
y = y_new

# If failed to converge well, do a coarse bracket scan and bisection
if not math.isfinite(y) or y <= 0.0 or abs(RHS(y) - A) > 1e-6:
# coarse grid excluding y near 1
grid1 = [1e-10 + i*(0.98 - 1e-10)/200 for i in range(201)]
grid2 = [1.02 + i*(3.0 - 1.02)/200 for i in range(201)]
grid = grid1 + grid2
vals = [RHS(yg) - A for yg in grid]
br_lo, br_hi = None, None
for i in range(len(grid)-1):
if vals[i] == 0.0:
y = grid[i]
break
if vals[i]*vals[i+1] < 0.0:
br_lo, br_hi = grid[i], grid[i+1]
break
if br_lo is not None:
for _ in range(60):
mid = 0.5*(br_lo + br_hi)
fm = RHS(mid) - A
if abs(fm) < tol:
y = mid
break
if fm* (RHS(br_lo) - A) > 0.0:
br_lo = mid
else:
br_hi = mid
y = 0.5*(br_lo + br_hi)

if y <= 0.0 or not math.isfinite(y):
return 1.0 # fallback to ideal in pathological cases

Z = A / y
return Z
def main():
print("Z-factor via Sutton + Wichert–Aziz + Hall–Yarborough")
try:
P_psia = float(input("Enter pressure, psia: ").strip())
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
Ppr = P_psia / Ppc_corr_psia

Z = Z_HallYarborough(Tpr, Ppr)

print("\n--- Results (Hall–Yarborough) ---")
print(f"Inputs: P = {P_psia:.3f} psia, T = {T_F:.3f} F ({TR:.3f} R), gg = {gg:.5f}, "
f"yH2S={yH2S:.5f}, yCO2={yCO2:.5f}, yN2={yN2:.5f}")
print(f"Pseudocriticals (Sutton sweet): Tpc = {Tpc_R:.3f} R, Ppc = {Ppc_psia:.3f} psia")
print(f"Wichert–Aziz epsilon: eps = {eps_R:.3f} R")
print(f"Pseudocriticals (sour corrected): Tpc' = {Tpc_corr_R:.3f} R, Ppc' = {Ppc_corr_psia:.3f} psia")
print(f"Pseudoreduced properties: Tpr = {Tpr:.6f}, Ppr = {Ppr:.6f}")
print(f"Z-factor (Hall–Yarborough): Z = {Z:.6f}")
if name == "main":
main()
