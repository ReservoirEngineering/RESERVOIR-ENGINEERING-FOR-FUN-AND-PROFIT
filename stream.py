import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull

def build_wells(x, y):
    # Primary injector and producer
    inj = [(x, y)]
    prod = [(y, x)]
    # Image wells not used for plotting, still present for velocity calculations
    inj += [(x, -y), (-x, y), (-x, -y)]
    prod += [(-y, x), (y, -x), (-y, -x)]
    return inj, prod

def velocity_from_well(px, py, wx, wy, qft3d, h, phi, sign):
    dx = px - wx
    dy = py - wy
    r2 = dx*dx + dy*dy
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
    pts = [(x_inj, y_inj + r0)]
    angles_deg = list(range(0, 360, 10))
    angles_deg = [a for a in angles_deg if a != 90]
    for a in angles_deg:
        theta = math.radians(a)
        px = x_inj + r0 * math.cos(theta)
        py = y_inj + r0 * math.sin(theta)
        pts.append((px, py))
    return np.array(pts, dtype=float)

def order_points_cyclic(points):
    pts = np.array(points, dtype=float)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return pts[order]

def draw_wells(ax, injectors, producers, inj_circle_r=15.0, prod_circle_r=15.0):
    # Only plot the main injector and producer wells
    main_inj = [injectors[0]]
    main_prod = [producers[0]]
    for (x, y) in main_inj:
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
    for (x, y) in main_prod:
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
    front_every_n_steps=25,
    include_captured_in_front=False
):
    injectors, producers = build_wells(x, y)
    primary_producer = (y, x)
    bbl_to_ft3 = 5.615
    qft3d = q_bbl_per_day * bbl_to_ft3
    particles = make_initial_particles(x, y, r0=1.0)  # (36,2)
    n_particles = particles.shape[0]
    traj_x = [[] for _ in range(n_particles)]
    traj_y = [[] for _ in range(n_particles)]
    active = np.array([True] * n_particles, dtype=bool)
    captured_time = np.full(n_particles, np.nan)
    times = []
    frac_captured = []
    front_polylines = []
    front_times = []
    t = 0.0
    step = 0
    captured_count = 0
    for i in range(n_particles):
        traj_x[i].append(particles[i, 0])
        traj_y[i].append(particles[i, 1])
    times.append(t)
    frac_captured.append(captured_count / n_particles)
    if step % front_every_n_steps == 0:
        if include_captured_in_front:
            pts = particles.copy()
        else:
            pts = np.array([particles[i] for i in range(n_particles) if active[i]], dtype=float)
        if pts.shape[0] >= 3:
            front_polylines.append(order_points_cyclic(pts))
            front_times.append(t)
    while (captured_count < capture_target) and (step < max_steps):
        step += 1
        t += dt_days
        for i in range(n_particles):
            if not active[i]:
                continue
            px, py = particles[i]
            vx, vy = total_velocity(px, py, injectors, producers, qft3d, thickness_ft, phi)
            px_new = px + vx * dt_days
            py_new = py + vy * dt_days
            particles[i, 0] = px_new
            particles[i, 1] = py_new
            traj_x[i].append(px_new)
            traj_y[i].append(py_new)
            dx = px_new - primary_producer[0]
            dy = py_new - primary_producer[1]
            if (dx*dx + dy*dy) <= (capture_radius_ft * capture_radius_ft):
                active[i] = False
                captured_count += 1
                captured_time[i] = t
        times.append(t)
        frac_captured.append(captured_count / n_particles)
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

def get_convex_hull(points):
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return None

def animate_streamlines_and_recovery(sim_data, frame_modulus=5, interval=70):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    injectors, producers = sim_data["injectors"], sim_data["producers"]
    primary_producer = sim_data["primary_producer"]
    traj_x, traj_y = sim_data["traj_x"], sim_data["traj_y"]
    times, frac_captured = sim_data["times"], sim_data["frac_captured"]
    captured_time = sim_data["captured_time"]
    n_particles = len(traj_x)
    max_frames = len(times)

    ax1 = axs[0]
    draw_wells(ax1, injectors, producers)
    scatters = [ax1.plot([], [], '-', lw=1.3, alpha=0.94)[0] for _ in range(n_particles)]
    last_pts = ax1.scatter([], [], c='k', s=18, zorder=6)
    ax1.set_aspect("equal")
    ax1.set_title("Streamlines/Particle Tracking")
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 700)

    ax2 = axs[1]
    draw_wells(ax2, injectors, producers)
    hull_poly, = ax2.plot([], [], 'b-', lw=3, alpha=0.5)
    active_pts = ax2.scatter([], [], c='b', s=20, zorder=7)
    ax2.set_aspect("equal")
    ax2.set_title("Current Injection Front (Convex Hull)")
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 700)

    ax3 = axs[2]
    line, = ax3.plot([], [], 'tab:orange', lw=2)
    ax3.set_ylim(-0.04, 1.04)
    ax3.set_xlim(0, times[-1] * 1.01)
    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Fraction captured")
    ax3.set_title("Fraction of Particles Captured")

    def update(frame):
        idx = frame * frame_modulus
        if idx >= max_frames:
            idx = max_frames - 1
        # Panel 1
        for i, (xs, ys) in enumerate(zip(traj_x, traj_y)):
            xs_show = xs[:min(idx + 1, len(xs))]
            ys_show = ys[:min(idx + 1, len(ys))]
            scatters[i].set_data(xs_show, ys_show)
        px_now = [traj_x[i][min(idx, len(traj_x[i])-1)] for i in range(n_particles)]
        py_now = [traj_y[i][min(idx, len(traj_y[i])-1)] for i in range(n_particles)]
        xs_active = []
        ys_active = []
        for i in range(n_particles):
            if np.isnan(captured_time[i]) or times[idx] < captured_time[i]:
                xs_active.append(px_now[i])
                ys_active.append(py_now[i])
        curr_pts = np.c_[xs_active, ys_active]
        last_pts.set_offsets(curr_pts)
        # Panel 2
        hull = get_convex_hull(curr_pts)
        if hull is not None:
            closed_hull = np.vstack([hull, hull[0]])
            hull_poly.set_data(closed_hull[:,0], closed_hull[:,1])
        else:
            hull_poly.set_data([], [])
        active_pts.set_offsets(curr_pts)
        # Panel 3
        line.set_data(times[:idx+1], frac_captured[:idx+1])
        ax1.set_title(f"Streamlines/Tracking\nTime = {times[idx]:.2f} days")
        ax2.set_title(f"Current Injection Front\nTime = {times[idx]:.2f} days\n({len(xs_active)} particles remain)")
        ax3.set_title(f"Fraction Captured\n{frac_captured[idx]*100:.1f}% at {times[idx]:.2f} days")
        return scatters + [last_pts, hull_poly, active_pts, line]
    num_frames = max_frames // frame_modulus + 1
    anim = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim

if __name__ == "__main__":
    # Simulation parameters
    x = 467.0
    y = 125.0
    q_bbl_per_day = 1000.0
    thickness_ft = 25.0
    phi = 0.2
    dt_days = 0.05
    capture_radius_ft = 2.0
    capture_target = 30
    max_steps = 20000
    front_every_n_steps = 500
    include_captured_in_front = False

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

    animate_streamlines_and_recovery(sim, frame_modulus=5, interval=70)