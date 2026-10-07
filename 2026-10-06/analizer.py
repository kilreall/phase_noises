import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import allantools
from matplotlib import rcParams

# plot settings
rcParams['font.size'] = 14
plt.rcParams["font.family"] = "Century Gothic"
plt.rcParams['savefig.dpi'] = 300
rcParams["legend.frameon"] = True

dt = 0.120 + 0.045
T = 0.010
k = 4*np.pi/780.24e-9
g_span = 2*np.pi/(k*T**2)*1e5
print(g_span)


def fit_func(t, A, B, phi):
    return A*np.cos(2*np.pi*T**2*t + phi)+B


cut = 6500
delay = 120
p_noise = {'int_clk': None,
           'ext_clk': None}
f_offset = {'int_clk': None,
            'ext_clk': None}
f_offset['int_clk'], p_noise['int_clk'] = np.loadtxt(f'tr_dat0022.csv', delimiter=',', unpack=True, skiprows=1)
f_offset['ext_clk'], p_noise['ext_clk'] = np.loadtxt(f'tr_dat0012.csv', delimiter=',', unpack=True, skiprows=1)

# plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(f_offset['int_clk'], p_noise['int_clk'], '-', lw=1, c='tab:blue', label='Internal Clock')
ax.plot(f_offset['ext_clk'], p_noise['ext_clk'], '-', lw=1, c='tab:red', label='External Clock (SRS+Rb Clock')

ax.grid(True)
ax.legend(loc='upper right')
ax.set_xscale('log')
ax.set_xlim([f_offset['int_clk'][0], f_offset['int_clk'][-1]])
ax.set_xlabel(r'Frequency Offset, [Hz]')
ax.set_ylabel(r'Phase Noise, [dBc/Hz]')


ax.set_title(r'Advantex Phase Noise (6.834 GHz)')
plt.tight_layout()
# plt.savefig(f'fig_vibrational_noise_{delay}.png')
plt.show()
