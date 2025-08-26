import numpy as np
import matplotlib.pyplot as plt

# Newman (1973) coefficients
a_ls = 0.853531          # Limestone
b_ls = 1.07538
c_ls = 2.30304e6
a_ss = 9.73e-5           # Sandstone
b_ss = 0.7
c_ss = 79.8181

# Valid ranges
phi_ls = np.linspace(0.02, 0.33, 200)
phi_ss = np.linspace(0.02, 0.23, 200)
phi_hall = np.linspace(0.05, 0.3, 200)

# Compute Newman limestone and sandstone
cf_ls = a_ls / ((1 + b_ls * c_ls * phi_ls) ** (1 / b_ls))
cf_ss = a_ss / ((1 + b_ss * c_ss * phi_ss) ** (1 / b_ss))

# Compute Hall (1953) for comparable range
cr = 1.87e-6 * phi_hall ** (-0.415)

# Convert porosity to %
phi_ls_percent = phi_ls * 100
phi_ss_percent = phi_ss * 100
phi_hall_percent = phi_hall * 100

# Convert compressibility to units of 10^{-6} psi^{-1}
cf_ls_plot = cf_ls / 1e-6
cf_ss_plot = cf_ss / 1e-6
cr_plot = cr / 1e-6

plt.figure(figsize=(7, 5))

# Limestone—solid line
plt.plot(phi_ls_percent, cf_ls_plot, 'k-', label='Newman Limestone')

# Sandstone—dashed line
plt.plot(phi_ss_percent, cf_ss_plot, 'k--', label='Newman Sandstone')

# Hall—dash-dot
plt.plot(phi_hall_percent, cr_plot, 'k-.', label='Hall (1953)')

plt.xlabel('Porosity (%)')
plt.ylabel('Rock Compressibility $(10^{-6}\\,\mathrm{psi}^{-1})$')
plt.xlim(0,33)
plt.ylim(0, max(cf_ls_plot.max(), cf_ss_plot.max(), cr_plot.max()) * 1.1)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)

# Add label box in top right of the axes, INSIDE the plot
# Adjust box position as needed: 23, y = maximum * 0.72, box size
label_text = (
    '–––––––––  Newman Limestone\n'
    '– – – – –  Newman Sandstone\n'
    '–·–·–·–·–  Hall (1953)'
)

plt.text(
    23, max(cf_ls_plot.max(), cf_ss_plot.max(), cr_plot.max()) * 0.72,
    label_text,
    fontsize=10, ha='left', va='top',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
)

plt.tight_layout()
plt.show()
