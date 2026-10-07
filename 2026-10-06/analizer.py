import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import allantools
from matplotlib import rcParams
from scipy.integrate import trapezoid


# plot settings
rcParams['font.size'] = 14
plt.rcParams["font.family"] = "Century Gothic"
plt.rcParams['savefig.dpi'] = 300
rcParams["legend.frameon"] = True



def fit_func(alp, T, A, B, ph):
    return A + B*np.cos(2*np.pi*T**2*alp - ph)

def H_w(w):
    return 4/(1-(w/W)**2)*np.sin(w*T/2)*(np.sin(w*(T-2*tau)/2) + w/W*np.cos(w*T/2))
H_w_vev = np.vectorize(H_w)

def count_phase(f_offset, L_dBc):

    f = np.asarray(f_offset, dtype=float)
    L = np.asarray(L_dBc, dtype=float)

    # сортировка и ограничение полосы
    idx = np.argsort(f)
    f, L = f[idx], L[idx]
    mask = np.ones_like(f, dtype=bool)
    if f_min is not None:
        mask &= f >= f_min
    if f_max is not None:
        mask &= f <= f_max
    f, L = f[mask], L[mask]

    S_ph = 2 * 10**(L / 10)           # рад^2/Гц
    H2 = H_w(2*np.pi*f)**2             # |H(2πf)|^2
    phase_integral = trapezoid(H2 * S_ph, f)
    return np.sqrt(phase_integral)


T_cycle = 0.120 + 0.045
tau = 5e-6
f_max = 1/tau
W = np.pi/2/tau
T = 0.010
f_min = 1/T
k_eff = 4*np.pi/780.24e-9
g_span = 2*np.pi/(k_eff*T**2)*1e5
print(f"g span = {round(g_span)} mGal")


cut = 6500
delay = 120
p_noise = {'int_clk': None,
           'ext_clk': None}
f_offset = {'int_clk': None,
            'ext_clk': None}
f_offset['int_clk'], p_noise['int_clk'] = np.loadtxt(r'2026-10-06\tr_dat0022.csv', delimiter=',', unpack=True, skiprows=1)
f_offset['ext_clk'], p_noise['ext_clk'] = np.loadtxt(r'2026-10-06\tr_dat0012.csv', delimiter=',', unpack=True, skiprows=1)

# plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(f_offset['int_clk'], p_noise['int_clk'], '-', lw=1, c='tab:blue', label='Внутренний часы advantex')
ax.plot(f_offset['ext_clk'], p_noise['ext_clk'], '-', lw=1, c='tab:red', label='Внешние Rb часы (SRS)')

ax.grid(True)
ax.legend(loc='upper right')
ax.set_xscale('log')
ax.set_xlim([f_offset['int_clk'][0], f_offset['int_clk'][-1]])
ax.set_xlabel(r'f, [Гц]')
ax.set_ylabel(r'L(f), [дБн/Гц]')


#ax.set_title(r'Advantex Phase Noise (6.834 GHz)')
plt.tight_layout()
# plt.savefig(f'fig_vibrational_noise_{delay}.png')

phase_int_clk = count_phase(f_offset['int_clk'], p_noise['int_clk'])
phase_ext_clk = count_phase(f_offset['ext_clk'], p_noise['ext_clk'])

print(f"Internal clock phase noise = {phase_int_clk*1e3} mrad")
print(f"Internal clock gravity noise = {phase_int_clk/k_eff/T/T*1e5} mGal")
print(f"Enternal clock phase noise = {phase_ext_clk*1e3} mrad")
print(f"Enternal clock gravity noise = {phase_ext_clk/k_eff/T/T*1e5} mGal")

plt.show()
