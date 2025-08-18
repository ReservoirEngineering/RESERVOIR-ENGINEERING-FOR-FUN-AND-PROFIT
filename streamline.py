# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

def build_wells(x, y):
    # Primary injector and producer
    inj = [(x, y)]
    prod = [(y, x)]
    # Image injectors for no-flow boundaries along x=0 and y=0
    inj += [(x, -y), (-x, y), (-x, -y)]
    # Image producers
    prod += [(-y, x), (y, -x), (-y, -x)]
    return inj, prod

def velocity_from_well(px, py, wx, wy, qft3d, h, phi, sign):
    # sign = +1 for injector (source), -1 for producer (sink)
    dx = px - wx
    dy = py - wy
    r2 = dx * dx + dy * dy
    if r2 < 1e-10:
        r2 = 1e-10
    coeff = sign * qft3d / (2.0 * math.pi * h * phi)
    vx = coeff * dx / r2
    vy = coeff * dy / r2
    return vx, vy

def total_velocity(px, py, injectors, producers, qft3d, h, phi):
    vx = 0.0
    vy = 0.0
    for (wx, wy) in injectors:
        dvx, dvy = velocity_from_well(px, py, wx, wy, qft3d, h, phi, sign=+1)
        vx += dvx
        vy += dvy
    for (wx, wy) in producers:
        dvx, dvy = velocity_from_well(px, py, wx, wy, qft3d, h, phi, sign=-1)
        vx += dvx
        vy += dvy
    return vx, vy

def make_initial_particles(x_inj, y_inj, r0=1.0):
    # 36 particles: one directly above + 35 at 10° intervals (excluding 90° to avoid duplicate)
    pts = [(x_inj, y_inj + r0)]
    angles_deg = list(range(0, 360, 10))  # 0..350
    angles_deg = [a for a in angles_deg if a != 90]
    for a in angles_deg:
        theta = math.radians(a)
        px = x_inj + r0 * math.cos(theta)
        py = y_inj + r0 * math.sin(theta)
        pts.append((px, py))
    return np.array(pts, dtype=float)

def order_points_cyclic(points):
    # Order points by angle around the centroid for a sensible polyline
    pts = np.array(points, dtype=float)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return pts[order]

def draw_wells(ax, injectors, producers, inj_circle_r=15.0, prod_circle_r=15.0):
    # Draw axes (no-flow boundaries)
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.axvline(0, color="k", lw=1.0, alpha=0.6)
    # Draw injectors: circle + arrow at 45°
    for (x, y) in injectors:
        circ = plt.Circle((x, y), inj_circle_r, facecolor="none", edgecolor="tab:green", lw=1.5)
        ax.add_patch(circ)
        length_out = 3.0 * inj_circle_r
        dx = length_out / math.sqrt(2.0)
        dy = length_out / math.sqrt(2.0)
        start_shift = inj_circle_r / math.sqrt(2.0)
        x0 = x - start_shift
        y0 = y - start_shift
        ax.annotate(
            "",
            xy=(x0 + dx, y0 + dy),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.5),
        )
    # Draw producers: open circles
    for (x, y) in producers:
        circ = plt.Circle((x, y), prod_circle_r, facecolor="none", edgecolor="tab:red", lw=1.5)
        ax.add_patch(circ)

def simulate_streamlines(
    x=467.0,
    y=125.0,
    q_bbl_per_day=1000.0,
    thickness_ft=25.0,
    phi=0.2,
    dt_days=0.05,
    capture_radius_ft=2.0,
    max_steps=10000,
    capture_target=30,
    front_every_n_steps=25,   # snapshot frequency for front lines
    include_captured_in_front=False  # True to include captured points in front lines
):
    injectors, producers = build_wells(x, y)
    primary_producer = (y, x)

    # Convert to ft^3/day
    bbl_to_ft3 = 5.615
    qft3d = q_bbl_per_day * bbl_to_ft3

    # Initialize particles
    particles = make_initial_particles(x, y, r0=1.0)  # (36,2)
    n_particles = particles.shape[0]

    # Trajectories
    traj_x = [[] for _ in range(n_particles)]
    traj_y = [[] for _ in range(n_particles)]
    active = np.array([True] * n_particles, dtype=bool)
    captured_time = np.full(n_particles, np.nan)

    # Time tracking
    times = []
    frac_captured = []

    # Front snapshots (polylines and times)
    front_polylines = []  # list of np.array shape (k,2), ordered cyclically
    front_times = []      # list of times corresponding to polylines

    # Init record
    t = 0.0
    step = 0
    captured_count = 0
    for i in range(n_particles):
        traj_x[i].append(particles[i, 0])
        traj_y[i].append(particles[i, 1])
    times.append(t)
    frac_captured.append(captured_count / n_particles)

    # Initial front snapshot at step 0
    if step % front_every_n_steps == 0:
        if include_captured_in_front:
            pts = particles.copy()
        else:
            pts = np.array([particles[i] for i in range(n_particles) if active[i]], dtype=float)
        if pts.shape[0] >= 3:
            front_polylines.append(order_points_cyclic(pts))
            front_times.append(t)

    # Simulation loop
    while (captured_count < capture_target) and (step < max_steps):
        step += 1
        t += dt_days

        for i in range(n_particles):
            if not active[i]:
                continue
            px, py = particles[i]
            vx, vy = total_velocity(px, py, injectors, producers, qft3d, thickness_ft, phi)

            # Forward Euler step
            px_new = px + vx * dt_days
            py_new = py + vy * dt_days
            particles[i, 0] = px_new
            particles[i, 1] = py_new

            # Record trajectory
            traj_x[i].append(px_new)
            traj_y[i].append(py_new)

            # Capture check (primary producer)
            dx = px_new - primary_producer[0]
            dy = py_new - primary_producer[1]
            if (dx * dx + dy * dy) <= (capture_radius_ft * capture_radius_ft):
                active[i] = False
                captured_count += 1
                captured_time[i] = t

        times.append(t)
        frac_captured.append(captured_count / n_particles)

        # Front snapshot every N steps
        if step % front_every_n_steps == 0:
            if include_captured_in_front:
                pts = particles.copy()
            else:
                pts = np.array([particles[i] for i in range(n_particles) if active[i]], dtype=float)
            if pts.shape[0] >= 3:
                front_polylines.append(order_points_cyclic(pts))
                front_times.append(t)

        if captured_count >= n_particles:
            break

    sim_data = {
        "injectors": injectors,
        "producers": producers,
        "primary_producer": primary_producer,
        "traj_x": traj_x,
        "traj_y": traj_y,
        "times": np.array(times),
        "frac_captured": np.array(frac_captured),
        "captured_time": captured_time,
        "final_particles": particles.copy(),
        "active": active.copy(),
        "front_polylines": front_polylines,
        "front_times": np.array(front_times),
        "dt_days": dt_days,
        "front_every_n_steps": front_every_n_steps,
    }
    return sim_data

def plot_streamlines(sim_data, title="Particle Streamlines with Image Wells"):
    injectors = sim_data["injectors"]
    producers = sim_data["producers"]
    traj_x = sim_data["traj_x"]
    traj_y = sim_data["traj_y"]

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_wells(ax, injectors, producers)

    # Plot trajectories
    cmap = plt.get_cmap("tab20")
    for i, (xs, ys) in enumerate(zip(traj_x, traj_y)):
        color = cmap(i % 20)
        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.95)

    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    all_x = []
    all_y = []
    for xs, ys in zip(traj_x, traj_y):
        all_x.extend(xs)
        all_y.extend(ys)
    for (xw, yw) in injectors + producers:
        all_x.append(xw)
        all_y.append(yw)
    if len(all_x) > 0:
        pad = 50.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    return fig, ax

def plot_front_lines(sim_data, title="Injection Front (every N steps)", alpha=0.9):
    injectors = sim_data["injectors"]
    producers = sim_data["producers"]
    polylines = sim_data["front_polylines"]
    times = sim_data["front_times"]
    captured_time = sim_data["captured_time"]

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_wells(ax, injectors, producers)

    # Color by time for clarity (lines only, no markers)
    if len(times) > 0:
        tmin, tmax = float(times.min()), float(times.max())
    else:
        tmin, tmax = 0.0, 1.0
    cmap = plt.get_cmap("viridis")

    # Stop plotting once the first particle is captured
    if np.all(np.isnan(captured_time)):
        t_first_capture = float("inf")
    else:
        t_first_capture = np.nanmin(captured_time)

    for poly, t in zip(polylines, times):
        if t > t_first_capture:
            break
        closed = np.vstack([poly, poly[0]])  # close the loop
        c = (t - tmin) / (tmax - tmin + 1e-12)
        ax.plot(closed[:, 0], closed[:, 1], '-', lw=2.0, color=cmap(c), alpha=alpha)

    ax.set_title(title + f" (N={sim_data['front_every_n_steps']}, dt={sim_data['dt_days']:.3f} d)")
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_aspect("equal", adjustable="box")

    # Fit extents to polylines and wells
    xs = []
    ys = []
    for poly in polylines:
        xs.extend(poly[:, 0])
        ys.extend(poly[:, 1])
    for (xw, yw) in injectors + producers:
        xs.append(xw)
        ys.append(yw)
    if xs:
        pad = 50.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    return fig, ax

def plot_fraction_captured(sim_data, title="Fraction of Particles Captured vs Time"):
    times = sim_data["times"]
    frac = sim_data["frac_captured"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, frac, lw=2.0, color="tab:orange")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fraction captured")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    return fig, ax

if __name__ == "__main__":
    # Given starting parameters
    x = 467.0
    y = 125.0
    q_bbl_per_day = 1000.0
    thickness_ft = 25.0
    phi = 0.2

    dt_days = 0.05
    capture_radius_ft = 2.0
    capture_target = 30
    max_steps = 20000

    # Front snapshot frequency set to 25 steps
    front_every_n_steps = 500
    include_captured_in_front = False  # change to True if you want all 36 included

    sim = simulate_streamlines(
        x=x,
        y=y,
        q_bbl_per_day=q_bbl_per_day,
        thickness_ft=thickness_ft,
        phi=phi,
        dt_days=dt_days,
        capture_radius_ft=capture_radius_ft,
        capture_target=capture_target,
        max_steps=max_steps,
        front_every_n_steps=front_every_n_steps,
        include_captured_in_front=include_captured_in_front
    )

    # Plots
    fig1, ax1 = plot_streamlines(sim, title="Streamlines: Injector-Producer with No-Flow Boundaries (Images)")
    fig2, ax2 = plot_front_lines(sim, title="Injection Front (lines only, every 25 steps)")

    # Example: dots-only front every 500 steps
    fig3, ax3 = plot_fraction_captured(sim, title="Fraction of 36 Particles Captured vs Time")

    # Ensure all plots show
    plt.show()
