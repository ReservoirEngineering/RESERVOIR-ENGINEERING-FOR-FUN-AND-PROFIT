import math

def exp_utility(x, a):
    return 1.0 - math.exp(-a * x)

def certainty_equivalent(EU, a):
    if a == 0.0:
        return EU  # interpreted as EMV
    return -math.log(1.0 - EU) / a

def evaluate_tree(a=0.0):
    p_high = 0.45
    payoff_high = 300.0 - 20.0  # develop after good test less appraisal cost
    payoff_low  =  40.0 - 20.0  # develop after poor test less appraisal cost

    p_up = 0.50
    payoff_up = 220.0
    payoff_down = -30.0

    if a == 0.0:
        emv_appraise = p_high * payoff_high + (1.0 - p_high) * payoff_low
        emv_noapp = p_up * payoff_up + (1.0 - p_up) * payoff_down
        return {"Appraise_EMV": emv_appraise, "NoAppraisal_EMV": emv_noapp}

    EU_appraise = p_high * exp_utility(payoff_high * 1e6, a) \
                + (1.0 - p_high) * exp_utility(payoff_low * 1e6, a)
    CE_appraise = certainty_equivalent(EU_appraise, a) / 1e6

    EU_noapp = p_up * exp_utility(payoff_up * 1e6, a) \
             + (1.0 - p_up) * exp_utility(payoff_down * 1e6, a)
    CE_noapp = certainty_equivalent(EU_noapp, a) / 1e6

    return {"Appraise_CE": CE_appraise, "NoAppraisal_CE": CE_noapp}

print(evaluate_tree(a=0.0))            # Risk neutral EMV
print(evaluate_tree(a=1.5e-9))         # Risk averse CE (in millions)
