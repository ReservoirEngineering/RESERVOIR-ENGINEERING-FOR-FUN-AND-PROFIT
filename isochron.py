import numpy as np
from scipy import interpolate
from scipy.stats import linregress
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: READ INPUT DATA
# =============================================================================

def read_isochron_data(filename):
    """Read reservoir and test parameters from isochron.dat"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    params = {
        'p_i': float(lines[0]),      # Initial pressure, psi
        'k': float(lines[1]),         # Permeability, md
        'h': float(lines[2]),         # Net pay thickness, ft
        'phi': float(lines[3]),       # Porosity, fraction
        'S_w': float(lines[4]),       # Water saturation, fraction
        'T_F': float(lines[5]),       # Temperature, °F
        'skin': float(lines[6]),      # Skin factor
        'r_e': float(lines[7]),       # External radius, ft
        'r_w': float(lines[8]),       # Wellbore radius, ft
        'q1': float(lines[9]),        # Flow rate 1, Mscf/D
        'q2': float(lines[10]),       # Flow rate 2, Mscf/D
        'q3': float(lines[11]),       # Flow rate 3, Mscf/D
        'q4': float(lines[12]),       # Flow rate 4, Mscf/D
        'q5': float(lines[13]),       # Flow rate 5 (extended), Mscf/D
        't1': float(lines[14]),       # Isochronal period, hours
        't2': float(lines[15]),       # Extended flow period, hours
    }
    
    # Convert temperature to Rankine
    params['T_R'] = params['T_F'] + 460.0
    
    # Gas saturation
    params['S_g'] = 1.0 - params['S_w']
    
    return params


def read_pvt_data(filename):
    """Read pressure, z-factor, and viscosity from muz.dat"""
    data = np.loadtxt(filename)
    return {
        'p': data[:, 0],      # Pressure, psia
        'z': data[:, 1],      # Z-factor
        'mu': data[:, 2]      # Viscosity, cp
    }


# =============================================================================
# SECTION 2: BUILD INTERPOLATION FUNCTIONS
# =============================================================================

def build_interpolators(pvt):
    """Create interpolation functions for z(p) and mu(p)"""
    z_interp = interpolate.interp1d(pvt['p'], pvt['z'], 
                                     kind='linear', fill_value='extrapolate')
    mu_interp = interpolate.interp1d(pvt['p'], pvt['mu'], 
                                      kind='linear', fill_value='extrapolate')
    return z_interp, mu_interp


def compute_dz_dp(p_array, z_interp):
    """
    Compute dz/dp numerically using central differences
    Returns interpolation function for dz/dp
    """
    dp = 0.1  # Small pressure increment for numerical derivative
    dz_dp_values = np.zeros_like(p_array)
    
    for i, p in enumerate(p_array):
        z_plus = z_interp(p + dp)
        z_minus = z_interp(p - dp)
        dz_dp_values[i] = (z_plus - z_minus) / (2 * dp)
    
    dz_dp_interp = interpolate.interp1d(p_array, dz_dp_values, 
                                         kind='linear', fill_value='extrapolate')
    return dz_dp_interp


def compute_gas_compressibility(p, z_interp, dz_dp_interp):
    """
    Calculate gas compressibility: c_g = 1/p - (1/z)(dz/dp)
    Returns c_g in psi^-1
    """
    z = z_interp(p)
    dz_dp = dz_dp_interp(p)
    c_g = 1.0/p - (1.0/z) * dz_dp
    return c_g


# =============================================================================
# SECTION 3: PSEUDOPRESSURE CALCULATIONS
# =============================================================================

def build_pseudopressure_table(pvt, z_interp, mu_interp, p_base=14.7):
    """
    Build pseudopressure table: m(p) = 2 * integral(p/(mu*z)) dp
    Uses trapezoidal integration
    Returns interpolation function m(p) and inverse p(m)
    """
    # Create fine pressure grid for integration
    p_min = pvt['p'].min()
    p_max = pvt['p'].max()
    p_fine = np.linspace(p_min, p_max, 5000)
    
    # Compute integrand: 2 * p / (mu * z)
    z_vals = z_interp(p_fine)
    mu_vals = mu_interp(p_fine)
    integrand = 2.0 * p_fine / (mu_vals * z_vals)
    
    # Integrate using cumulative trapezoidal rule
    m_vals = np.zeros_like(p_fine)
    for i in range(1, len(p_fine)):
        m_vals[i] = m_vals[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * (p_fine[i] - p_fine[i-1])
    
    # Create interpolation functions
    m_of_p = interpolate.interp1d(p_fine, m_vals, 
                                   kind='linear', fill_value='extrapolate')
    p_of_m = interpolate.interp1d(m_vals, p_fine, 
                                   kind='linear', fill_value='extrapolate')
    
    return m_of_p, p_of_m, p_fine, m_vals


# =============================================================================
# SECTION 4: BUILD RATE SCHEDULE (SUPERPOSITION)
# =============================================================================

def build_rate_schedule(params):
    """
    Build the rate change schedule for superposition
    
    Test sequence:
    0 -> t1:      Flow q1      (Δq = +q1 at t=0)
    t1 -> 2t1:    Shut-in      (Δq = -q1 at t=t1)
    2t1 -> 3t1:   Flow q2      (Δq = +q2 at t=2t1)
    3t1 -> 4t1:   Shut-in      (Δq = -q2 at t=3t1)
    4t1 -> 5t1:   Flow q3      (Δq = +q3 at t=4t1)
    5t1 -> 6t1:   Shut-in      (Δq = -q3 at t=5t1)
    6t1 -> 7t1:   Flow q4      (Δq = +q4 at t=6t1)
    7t1 -> 7t1+t2: Flow q5     (Δq = (q5-q4) at t=7t1)
    
    Returns list of (time, delta_q) tuples
    """
    t1 = params['t1']
    q1, q2, q3, q4, q5 = params['q1'], params['q2'], params['q3'], params['q4'], params['q5']
    
    rate_changes = [
        (0.0,    +q1),           # Start flow 1
        (t1,     -q1),           # Shut-in 1
        (2*t1,   +q2),           # Start flow 2
        (3*t1,   -q2),           # Shut-in 2
        (4*t1,   +q3),           # Start flow 3
        (5*t1,   -q3),           # Shut-in 3
        (6*t1,   +q4),           # Start flow 4
        (7*t1,   q5 - q4),       # Change to extended flow rate
    ]
    
    return rate_changes


def get_current_rate(t, rate_changes):
    """Calculate the current flow rate at time t"""
    q = 0.0
    for t_change, delta_q in rate_changes:
        if t >= t_change:
            q += delta_q
    return q


# =============================================================================
# SECTION 5: PRESSURE TRANSIENT CALCULATION
# =============================================================================

def calculate_delta_m(t, rate_changes, params, mu_avg, c_t_avg):
    """
    Calculate pseudopressure drop using superposition
    
    Δm = (1422 * T / (k * h)) * Σ Δq_j * F(t - t_j)
    
    where F(Δt) = 0.5 * ln(0.0002637 * k * Δt / (φ * μ * c_t * r_w^2)) + 0.80907 + s
    
    The constant 0.0002637 converts field units (k in md, t in hours, etc.)
    to dimensionless time.
    """
    if t <= 0:
        return 0.0
    
    T = params['T_R']
    k = params['k']
    h = params['h']
    phi = params['phi']
    r_w = params['r_w']
    skin = params['skin']
    
    # Coefficient outside summation
    coeff = 1422.0 * T / (k * h)
    
    # Unit conversion constant for dimensionless time
    # t_D = 0.0002637 * k * t / (phi * mu * c_t * r_w^2)
    conversion = 0.0002637 * k / (phi * mu_avg * c_t_avg * r_w**2)
    
    # Sum over all rate changes that have occurred
    delta_m = 0.0
    for t_j, delta_q_j in rate_changes:
        delta_t = t - t_j
        if delta_t > 0:
            t_D = conversion * delta_t
            # F function (dimensionless pressure + skin)
            F = 0.5 * np.log(t_D) + 0.80907 + skin
            delta_m += delta_q_j * F
    
    delta_m *= coeff
    return delta_m


# =============================================================================
# SECTION 6: MAIN SIMULATION
# =============================================================================

def run_simulation(params, pvt, dt=0.1):
    """
    Run the isochronal test simulation
    
    Parameters:
    -----------
    params : dict - Reservoir and test parameters
    pvt : dict - PVT data
    dt : float - Time step in hours
    
    Returns:
    --------
    results : dict containing time, pressure, rate arrays and key values
    """
    
    # Build interpolators
    z_interp, mu_interp = build_interpolators(pvt)
    dz_dp_interp = compute_dz_dp(pvt['p'], z_interp)
    
    # Build pseudopressure table
    m_of_p, p_of_m, p_table, m_table = build_pseudopressure_table(
        pvt, z_interp, mu_interp
    )
    
    # Initial pseudopressure
    p_i = params['p_i']
    m_i = m_of_p(p_i)
    
    # Calculate average properties at initial pressure
    mu_avg = mu_interp(p_i)
    c_g_avg = compute_gas_compressibility(p_i, z_interp, dz_dp_interp)
    c_t_avg = c_g_avg * params['S_g']  # Neglecting c_w and c_f
    
    print(f"Average properties at p_i = {p_i} psi:")
    print(f"  μ = {mu_avg:.6f} cp")
    print(f"  c_g = {c_g_avg:.6e} psi^-1")
    print(f"  c_t = {c_t_avg:.6e} psi^-1")
    print(f"  m(p_i) = {m_i:.2f} psi^2/cp")
    print()
    
    # Build rate schedule
    rate_changes = build_rate_schedule(params)
    
    # Define time array
    t1 = params['t1']
    t2 = params['t2']
    t_end = 7 * t1 + t2  # Total test duration
    
    # Create time array with specified resolution
    t_array = np.arange(0, t_end + dt, dt)
    
    # Ensure critical times are included
    critical_times = [
        t1,      # End of flow 1
        2*t1,    # End of shut-in 1
        3*t1,    # End of flow 2
        4*t1,    # End of shut-in 2
        5*t1,    # End of flow 3
        6*t1,    # End of shut-in 3
        7*t1,    # End of flow 4
        7*t1 + t2  # End of extended flow
    ]
    
    # Merge critical times into array
    t_array = np.unique(np.sort(np.concatenate([t_array, critical_times])))
    
    # Initialize output arrays
    n_points = len(t_array)
    p_wf = np.zeros(n_points)
    q_array = np.zeros(n_points)
    m_wf_array = np.zeros(n_points)
    
    # Run simulation
    print("Running simulation...")
    print("-" * 60)
    
    for i, t in enumerate(t_array):
        if t == 0:
            p_wf[i] = p_i
            q_array[i] = 0.0
            m_wf_array[i] = m_i
        else:
            # Calculate pseudopressure drop
            delta_m = calculate_delta_m(t, rate_changes, params, mu_avg, c_t_avg)
            
            # Calculate flowing pseudopressure
            m_wf = m_i - delta_m
            
            # Ensure m_wf doesn't go below minimum in table
            m_min = m_table.min()
            if m_wf < m_min:
                print(f"Warning: m(p_wf) = {m_wf:.2f} below table minimum at t = {t:.2f} hr")
                m_wf = m_min
            
            m_wf_array[i] = m_wf
            
            # Convert back to pressure
            p_wf[i] = p_of_m(m_wf)
            
            # Get current rate
            q_array[i] = get_current_rate(t, rate_changes)
    
    # Extract key values at end of each flow period
    key_times = [t1, 3*t1, 5*t1, 7*t1, 7*t1 + t2]
    key_rates = [params['q1'], params['q2'], params['q3'], params['q4'], params['q5']]
    key_pressures = []
    
    print("\n" + "=" * 60)
    print("KEY RESULTS - Flowing Pressures at End of Each Flow Period")
    print("=" * 60)
    print(f"{'Period':<12} {'Time (hr)':<12} {'Rate (Mscf/D)':<15} {'Pwf (psi)':<12}")
    print("-" * 60)
    
    for j, (t_key, q_key) in enumerate(zip(key_times, key_rates)):
        idx = np.argmin(np.abs(t_array - t_key))
        p_key = p_wf[idx]
        key_pressures.append(p_key)
        period_name = f"Flow {j+1}" if j < 4 else "Extended"
        print(f"{period_name:<12} {t_key:<12.1f} {q_key:<15.0f} {p_key:<12.2f}")
    
    print("=" * 60)
    
    # Also report shut-in pressures
    shutin_times = [2*t1, 4*t1, 6*t1]
    print("\nShut-in Pressures (end of each buildup):")
    print("-" * 40)
    for j, t_si in enumerate(shutin_times):
        idx = np.argmin(np.abs(t_array - t_si))
        print(f"  Shut-in {j+1} at t = {t_si:.1f} hr: Pws = {p_wf[idx]:.2f} psi")
    
    # Package results
    results = {
        't': t_array,
        'p_wf': p_wf,
        'q': q_array,
        'm_wf': m_wf_array,
        'key_times': key_times,
        'key_rates': key_rates,
        'key_pressures': key_pressures,
        'params': params
    }
    
    return results


# =============================================================================
# SECTION 7: DELIVERABILITY ANALYSIS (C, n, AOF)
# =============================================================================

def analyze_deliverability(results):
    """
    Perform deliverability analysis:
    1. Fit C and n from isochronal points (q1-q4)
    2. Calculate stabilized C using extended flow point
    3. Calculate AOF
    
    Backpressure equation: q = C * (Pi^2 - Pwf^2)^n
    Rearranged: (Pi^2 - Pwf^2) = (q/C)^(1/n)
    
    For log-log plot: log(Pi^2 - Pwf^2) = (1/n)*log(q) - (1/n)*log(C)
    Or: log(q) = log(C) + n*log(Pi^2 - Pwf^2)
    """
    
    p_i = results['params']['p_i']
    p_atm = 14.7  # Atmospheric pressure for AOF
    
    # Isochronal points (first 4 flow periods)
    q_iso = np.array(results['key_rates'][:4])
    p_iso = np.array(results['key_pressures'][:4])
    delta_p2_iso = p_i**2 - p_iso**2
    
    # Extended flow point
    q_ext = results['key_rates'][4]
    p_ext = results['key_pressures'][4]
    delta_p2_ext = p_i**2 - p_ext**2
    
    # Fit isochronal line: log(q) = log(C) + n*log(delta_p2)
    log_q_iso = np.log10(q_iso)
    log_dp2_iso = np.log10(delta_p2_iso)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_dp2_iso, log_q_iso)
    
    n = slope
    C_iso = 10**intercept
    
    print("\n" + "=" * 60)
    print("DELIVERABILITY ANALYSIS")
    print("=" * 60)
    print("\nIsochronal Line Fit (from q1, q2, q3, q4):")
    print(f"  n = {n:.4f}")
    print(f"  C_isochronal = {C_iso:.6e} Mscf/D/psi^(2n)")
    print(f"  R² = {r_value**2:.6f}")
    
    # Calculate stabilized C using same n, through extended point
    # q_ext = C_stab * (delta_p2_ext)^n
    # C_stab = q_ext / (delta_p2_ext)^n
    C_stab = q_ext / (delta_p2_ext**n)
    
    print("\nStabilized (Extended Flow) Analysis:")
    print(f"  Using n = {n:.4f} (from isochronal fit)")
    print(f"  Extended flow point: q = {q_ext:.0f} Mscf/D, Pwf = {p_ext:.2f} psi")
    print(f"  C_stabilized = {C_stab:.6e} Mscf/D/psi^(2n)")
    
    # Calculate AOF using stabilized C
    # AOF = C_stab * (Pi^2 - Patm^2)^n
    delta_p2_aof = p_i**2 - p_atm**2
    AOF = C_stab * (delta_p2_aof**n)
    
    print("\nAbsolute Open Flow (AOF):")
    print(f"  At Pwf = {p_atm} psia (atmospheric)")
    print(f"  Pi² - Pwf² = {delta_p2_aof:.2f} psi²")
    print(f"  AOF = {AOF:.0f} Mscf/D")
    print("=" * 60)
    
    # Store analysis results
    analysis = {
        'n': n,
        'C_iso': C_iso,
        'C_stab': C_stab,
        'r_squared': r_value**2,
        'AOF': AOF,
        'q_iso': q_iso,
        'p_iso': p_iso,
        'delta_p2_iso': delta_p2_iso,
        'q_ext': q_ext,
        'p_ext': p_ext,
        'delta_p2_ext': delta_p2_ext,
        'delta_p2_aof': delta_p2_aof,
        'p_i': p_i,
        'p_atm': p_atm
    }
    
    return analysis


# =============================================================================
# SECTION 8: PLOTTING
# =============================================================================

def plot_results(results, analysis):
    """Create plots of the simulation results"""
    
    t = results['t']
    p_wf = results['p_wf']
    q = results['q']
    params = results['params']
    
    # =========================================================================
    # PLOT 1: Pressure and Rate vs Time
    # =========================================================================
    fig1, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1a: Pressure vs Time
    ax1 = axes[0]
    ax1.plot(t, p_wf, 'b-', linewidth=1.5)
    ax1.set_ylabel('Bottomhole Pressure (psi)', fontsize=12)
    ax1.set_title('Isochronal Gas Well Test - Pressure Response', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=params['p_i'], color='r', linestyle='--', alpha=0.5, label=f"Pi = {params['p_i']} psi")
    
    # Mark key points
    for j, (t_key, p_key) in enumerate(zip(results['key_times'], results['key_pressures'])):
        ax1.plot(t_key, p_key, 'ro', markersize=8)
        ax1.annotate(f'{p_key:.0f}', (t_key, p_key), textcoords="offset points", 
                    xytext=(5, 10), fontsize=9)
    
    ax1.legend(loc='upper right')
    
    # Plot 1b: Rate vs Time
    ax2 = axes[1]
    ax2.step(t, q, 'g-', linewidth=1.5, where='post')
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Flow Rate (Mscf/D)', fontsize=12)
    ax2.set_title('Flow Rate Schedule', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('isochronal_test_results.png', dpi=150)
    plt.show()
    
    # =========================================================================
    # PLOT 2: Log-Log Deliverability Plot (Square aspect ratio)
    # =========================================================================
    fig2, ax3 = plt.subplots(figsize=(8, 8))
    
    # Extract data from analysis
    q_iso = analysis['q_iso']
    delta_p2_iso = analysis['delta_p2_iso']
    q_ext = analysis['q_ext']
    delta_p2_ext = analysis['delta_p2_ext']
    delta_p2_aof = analysis['delta_p2_aof']
    AOF = analysis['AOF']
    n = analysis['n']
    C_iso = analysis['C_iso']
    C_stab = analysis['C_stab']
    
    # Plot isochronal points
    ax3.loglog(q_iso, delta_p2_iso, 'bo', markersize=12, label='Isochronal Points', zorder=5)
    
    # Plot extended flow point
    ax3.loglog(q_ext, delta_p2_ext, 'rs', markersize=14, label='Extended Flow Point', zorder=5)
    
    # Plot AOF point
    ax3.loglog(AOF, delta_p2_aof, 'g^', markersize=14, label=f'AOF = {AOF:.0f} Mscf/D', zorder=5)
    
    # Determine plot limits for square decades
    all_q = np.concatenate([q_iso, [q_ext, AOF]])
    all_dp2 = np.concatenate([delta_p2_iso, [delta_p2_ext, delta_p2_aof]])
    
    # Find range in log space
    log_q_min = np.floor(np.log10(all_q.min()))
    log_q_max = np.ceil(np.log10(all_q.max()))
    log_dp2_min = np.floor(np.log10(all_dp2.min()))
    log_dp2_max = np.ceil(np.log10(all_dp2.max()))
    
    # Make equal number of decades on each axis
    q_decades = log_q_max - log_q_min
    dp2_decades = log_dp2_max - log_dp2_min
    max_decades = max(q_decades, dp2_decades)
    
    # Expand the smaller range to match
    if q_decades < max_decades:
        extra = (max_decades - q_decades) / 2
        log_q_min -= extra
        log_q_max += extra
    if dp2_decades < max_decades:
        extra = (max_decades - dp2_decades) / 2
        log_dp2_min -= extra
        log_dp2_max += extra
    
    # Add some padding
    log_q_min -= 0.1
    log_q_max += 0.3  # Extra room for AOF
    log_dp2_min -= 0.1
    log_dp2_max += 0.1
    
    ax3.set_xlim(10**log_q_min, 10**log_q_max)
    ax3.set_ylim(10**log_dp2_min, 10**log_dp2_max)
    
    # Generate line data for isochronal fit (solid line)
    q_line = np.logspace(log_q_min, log_q_max, 100)
    # From q = C * (delta_p2)^n  =>  delta_p2 = (q/C)^(1/n)
    dp2_iso_line = (q_line / C_iso)**(1/n)
    ax3.loglog(q_line, dp2_iso_line, 'b-', linewidth=2, label=f'Isochronal: n={n:.3f}', zorder=3)
    
    # Generate line data for stabilized fit (dashed line through extended point to AOF)
    dp2_stab_line = (q_line / C_stab)**(1/n)
    ax3.loglog(q_line, dp2_stab_line, 'r--', linewidth=2, label=f'Stabilized: n={n:.3f}', zorder=3)
    
    # Labels and formatting
    ax3.set_xlabel('Flow Rate, q (Mscf/D)', fontsize=12)
    ax3.set_ylabel('$P_i^2 - P_{wf}^2$ (psi²)', fontsize=12)
    ax3.set_title('Isochronal Deliverability Plot (Log-Log)', fontsize=14)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10)
    
    # Force square aspect ratio
    ax3.set_aspect('equal', adjustable='box')
    
    # Add text annotation for AOF
    ax3.annotate(f'AOF = {AOF:.0f} Mscf/D', 
                xy=(AOF, delta_p2_aof), 
                xytext=(AOF*0.3, delta_p2_aof*1.5),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('deliverability_plot_loglog.png', dpi=150)
    plt.show()
    
    return fig1, fig2


# =============================================================================
# SECTION 9: OUTPUT TO FILE
# =============================================================================

def write_output(results, analysis, filename='isochronal_output.dat'):
    """Write simulation results to file"""
    
    t = results['t']
    p_wf = results['p_wf']
    q = results['q']
    
    with open(filename, 'w') as f:
        f.write("# Isochronal Gas Well Test - Simulated Pressure Response\n")
        f.write("# " + "=" * 50 + "\n")
        f.write(f"# Initial Pressure: {results['params']['p_i']} psi\n")
        f.write(f"# Temperature: {results['params']['T_F']} °F\n")
        f.write(f"# Permeability: {results['params']['k']} md\n")
        f.write(f"# Net Pay: {results['params']['h']} ft\n")
        f.write(f"# Skin Factor: {results['params']['skin']}\n")
        f.write("# " + "=" * 50 + "\n")
        f.write("#\n")
        f.write(f"# {'Time(hr)':<12} {'Rate(Mscf/D)':<15} {'Pwf(psi)':<12}\n")
        f.write("# " + "-" * 40 + "\n")
        
        for i in range(len(t)):
            f.write(f"  {t[i]:<12.2f} {q[i]:<15.1f} {p_wf[i]:<12.2f}\n")
    
    print(f"\nResults written to {filename}")
    
    # Write key values and analysis
    with open('key_values.dat', 'w') as f:
        f.write("# Isochronal Gas Well Test - Key Results\n")
        f.write("# " + "=" * 50 + "\n\n")
        
        f.write("# Flowing Pressures at End of Each Flow Period\n")
        f.write(f"# {'Time(hr)':<12} {'Rate(Mscf/D)':<15} {'Pwf(psi)':<12} {'Pi2-Pwf2(psi2)':<15}\n")
        f.write("# " + "-" * 55 + "\n")
        
        p_i = results['params']['p_i']
        for t_key, q_key, p_key in zip(results['key_times'], 
                                        results['key_rates'], 
                                        results['key_pressures']):
            dp2 = p_i**2 - p_key**2
            f.write(f"  {t_key:<12.1f} {q_key:<15.0f} {p_key:<12.2f} {dp2:<15.2f}\n")
        
        f.write("\n# " + "=" * 50 + "\n")
        f.write("# DELIVERABILITY ANALYSIS\n")
        f.write("# " + "=" * 50 + "\n\n")
        f.write(f"# Isochronal Fit:\n")
        f.write(f"#   n = {analysis['n']:.4f}\n")
        f.write(f"#   C_isochronal = {analysis['C_iso']:.6e} Mscf/D/psi^(2n)\n")
        f.write(f"#   R² = {analysis['r_squared']:.6f}\n\n")
        f.write(f"# Stabilized Fit:\n")
        f.write(f"#   n = {analysis['n']:.4f} (same as isochronal)\n")
        f.write(f"#   C_stabilized = {analysis['C_stab']:.6e} Mscf/D/psi^(2n)\n\n")
        f.write(f"# Absolute Open Flow:\n")
        f.write(f"#   AOF = {analysis['AOF']:.0f} Mscf/D\n")
    
    print("Key values and analysis written to key_values.dat")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the isochronal test simulation"""
    
    print("=" * 60)
    print("ISOCHRONAL GAS WELL TEST - PRESSURE RESPONSE GENERATOR")
    print("=" * 60)
    print()
    
    # Read input data
    print("Reading input files...")
    params = read_isochron_data('isochron.dat')
    pvt = read_pvt_data('muz.dat')
    
    # Print input summary
    print("\nReservoir Parameters:")
    print(f"  Initial Pressure: {params['p_i']} psi")
    print(f"  Permeability: {params['k']} md")
    print(f"  Net Pay: {params['h']} ft")
    print(f"  Porosity: {params['phi']}")
    print(f"  Water Saturation: {params['S_w']}")
    print(f"  Temperature: {params['T_F']} °F ({params['T_R']} °R)")
    print(f"  Skin Factor: {params['skin']}")
    print(f"  Drainage Radius: {params['r_e']} ft")
    print(f"  Wellbore Radius: {params['r_w']} ft")
    
    print("\nTest Parameters:")
    print(f"  q1 = {params['q1']} Mscf/D")
    print(f"  q2 = {params['q2']} Mscf/D")
    print(f"  q3 = {params['q3']} Mscf/D")
    print(f"  q4 = {params['q4']} Mscf/D")
    print(f"  q5 (extended) = {params['q5']} Mscf/D")
    print(f"  t1 (isochronal period) = {params['t1']} hours")
    print(f"  t2 (extended period) = {params['t2']} hours")
    
    t1 = params['t1']
    t2 = params['t2']
    print(f"\nTotal test duration: {7*t1 + t2} hours")
    print()
    

      # Run simulation
    results = run_simulation(params, pvt, dt=0.1)
    
    # Perform deliverability analysis
    analysis = analyze_deliverability(results)
    
    # Write output
    write_output(results, analysis)
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(results, analysis)
    
    print("\nSimulation complete!")
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()