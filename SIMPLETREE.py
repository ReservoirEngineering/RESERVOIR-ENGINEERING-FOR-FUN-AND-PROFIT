
import math

def real_option_defer(V0, I, sigma, r, T=2, dt=1.0):
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    V = [[0.0]*(t+1) for t in range(T+1)]
    V[0][0] = V0
    for t in range(1, T+1):
        for i in range(t+1):
            V[t][i] = V0 * (u**i) * (d**(t - i))

    C = [max(V[T][i] - I, 0.0) for i in range(T+1)]

    exercise = [[False]*(t+1) for t in range(T+1)]
    exercise[T] = [V[T][i] - I > 0.0 for i in range(T+1)]

    for t in range(T-1, -1, -1):
        C_next = []
        for i in range(t+1):
            hold = disc * (p * C[i+1] + (1 - p) * C[i])
            ex   = max(V[t][i] - I, 0.0)
            if ex > hold:
                C_next.append(ex)
                exercise[t][i] = True
            else:
                C_next.append(hold)
        C = C_next

    return {"OptionValue": C[0], "ExercisePolicy": exercise, "u": u, "d": d, "p": p}

res = real_option_defer(V0=320.0, I=300.0, sigma=0.35, r=0.05, T=2, dt=1.0)
print("Defer option value (million):", res["OptionValue"])
print("Risk-neutral up prob:", res["p"])
