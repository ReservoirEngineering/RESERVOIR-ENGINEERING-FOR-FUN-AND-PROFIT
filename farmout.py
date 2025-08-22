def farmout_emv(p_success=0.25, V_success=300.0, C_dry=30.0,
                wi_out=1.0, wi_after=0.5, carry_frac=0.80, carry_cap=20.0,
                p_overrun=0.2, overrun_excess=10.0):
    # Values in millions
    E_carry_cost = carry_frac * (C_dry + p_overrun * overrun_excess)
    E_carry_cost = min(E_carry_cost, carry_cap)
    EV_farmor = p_success * (wi_after * V_success - (1 - carry_frac) * (C_dry + p_overrun * overrun_excess)) \
              + (1 - p_success) * ( - (1 - carry_frac) * (C_dry + p_overrun * overrun_excess)
                                    + min(carry_frac * (C_dry + p_overrun * overrun_excess), carry_cap) )
    EV_farminee = p_success * (wi_after * V_success - min(carry_frac * (C_dry + p_overrun * overrun_excess), carry_cap)) \
                + (1 - p_success) * ( - min(carry_frac * (C_dry + p_overrun * overrun_excess), carry_cap) )
    return EV_farmor, EV_farminee

print(farmout_emv())
