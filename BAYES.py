def bayes_update(pH, sens, fpr):
    pS = sens * pH + fpr * (1 - pH)
    pH_S = sens * pH / pS
    pH_notS = (1 - sens) * pH / (1 - pS)
    return pS, pH_S, pH_notS

def ev_no_test(pH, VH, VnotH):
    return pH * VH + (1 - pH) * VnotH

def ev_with_test(pH, sens, fpr, VH, VnotH, Ctest):
    pS, pH_S, pH_notS = bayes_update(pH, sens, fpr)
    ev_dev_S = max(pH_S * VH + (1 - pH_S) * VnotH, 0.0)
    ev_dev_notS = max(pH_notS * VH + (1 - pH_notS) * VnotH, 0.0)
    return pS * ev_dev_S + (1 - pS) * ev_dev_notS - Ctest

def evpi(pH, VH, VnotH):
    # Perfect information lets you develop only in H
    return pH * VH + (1 - pH) * 0.0

pH = 0.35
VH, VnotH = 260.0, -50.0
sens, fpr = 0.80, 0.25
Ctest = 8.0

ev_nt = ev_no_test(pH, VH, VnotH)
ev_t = ev_with_test(pH, sens, fpr, VH, VnotH, Ctest)
print("EV without test:", ev_nt)
print("EV with test:", ev_t)
print("EVSI:", ev_t - ev_nt)
print("EVPI:", evpi(pH, VH, VnotH) - max(ev_nt, 0.0))
