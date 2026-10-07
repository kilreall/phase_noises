import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from pathlib import Path

# plot settings
rcParams['font.size'] = 14
rcParams['font.family'] = 'Century Gothic'
rcParams['savefig.dpi'] = 300
rcParams['legend.frameon'] = True

# ---------------- параметры ----------------
T_cycle = 0.120 + 0.045          # период цикла, с
tau = 5e-6                       # длительность pi/2-импульса, с
W = np.pi/2/tau                  # частота Раби, рад/с
T = 0.010                        # время свободной эволюции, с
k_eff = 4*np.pi/780.24e-9        # рад/м

f_min = 1/T                      # нижний предел интегрирования, Гц
f_max = 1/tau                    # верхний предел интегрирования, Гц
df = 1/(50*T)                    # шаг плотной сетки (|H|^2 осциллирует с периодом 1/(2T))

# ---- параметры сглаживания ----
SMOOTH = True
SMOOTH_KW = dict(
    pts_per_decade=100,   # плотность равномерной сетки по log10(f)
    hampel_win=7,         # окно фильтра выбросов, точек сетки
    hampel_sigma=3.0,     # порог выброса, в «сигмах» (MAD)
    sg_win=15,            # окно Савицкого-Голея, точек сетки (нечётное)
    sg_order=2,           # порядок полинома
)

g_span = 2*np.pi/(k_eff*T**2)*1e5
print(f"g span = {round(g_span)} mGal")


def H_w(w):
    """Весовая функция H(w). Особенность при w = W устранима, точку слегка сдвигаем."""
    w = np.asarray(w, dtype=float)
    w = np.where(np.abs(w/W - 1) < 1e-9, w*(1 + 1e-7), w)
    x = w/W
    return 4/(1 - x**2)*np.sin(w*T/2)*(np.sin(w*(T - 2*tau)/2) + x*np.cos(w*T/2))


def smooth_psd(f, L, pts_per_decade=100, hampel_win=7, hampel_sigma=3.0,
               sg_win=15, sg_order=2):
    """
    Сглаживание L(f) [дБн/Гц]: равномерная сетка по log10(f) -> фильтр Хампеля
    (убирает одиночные выбросы) -> Савицкий-Голей. f должна быть отсортирована, f > 0.
    Возвращает (f_сглаженная_сетка, L_сглаженная).
    """
    lf = np.log10(f)
    n = int(np.floor((lf[-1] - lf[0])*pts_per_decade)) + 1
    lg = np.linspace(lf[0], lf[-1], n)
    Lg = np.interp(lg, lf, L)

    # Хампель: точка - выброс, если отклонение от скользящей медианы > sigma * 1.4826 * MAD
    med = median_filter(Lg, size=hampel_win, mode='nearest')
    mad = 1.4826*median_filter(np.abs(Lg - med), size=hampel_win, mode='nearest')
    outlier = np.abs(Lg - med) > hampel_sigma*np.maximum(mad, 1e-3)
    Lc = np.where(outlier, med, Lg)
    print(f"  выбросов заменено: {outlier.sum()} из {len(Lg)}")

    # Савицкий-Голей (окно нечётное и не больше длины данных)
    win = min(sg_win, len(Lc) - (1 - len(Lc) % 2))
    win = win if win % 2 == 1 else win - 1
    Ls = savgol_filter(Lc, win, min(sg_order, win - 1), mode='interp')
    return 10**lg, Ls


def interpolate_psd(f_offset, L_dBc, f_min, f_max, df, smooth=False, **smooth_kw):
    """
    L(f) [дБн/Гц] -> (опционально сглаживание) -> плотная равномерная сетка fg [Гц]
    (+ узловые точки в полосе). Интерполяция линейная в дБ по log10(f).
    """
    f = np.asarray(f_offset, dtype=float)
    L = np.asarray(L_dBc, dtype=float)
    idx = np.argsort(f)
    f, L = f[idx], L[idx]
    keep = f > 0
    f, L = f[keep], L[keep]

    if smooth:
        f, L = smooth_psd(f, L, **smooth_kw)

    lo, hi = max(f_min, f[0]), min(f_max, f[-1])
    in_band = (f >= lo) & (f <= hi)
    fg = np.union1d(np.arange(lo, hi, df), np.concatenate(([lo, hi], f[in_band])))
    return fg, np.interp(np.log10(fg), np.log10(f), L)


def count_phase(fg, Lg):
    """sigma_phi [рад] = sqrt(int |H(2 pi f)|^2 S_phi(f) df) по интерполированным данным."""
    S_ph = 2*10**(Lg/10)                 # рад^2/Гц
    H2 = H_w(2*np.pi*fg)**2
    assert np.isfinite(H2).all(), "nan/inf в |H|^2"
    return np.sqrt(trapezoid(H2*S_ph, fg))


# ---------------- данные ----------------
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / '2026-10-06'
if not data_dir.is_dir():          # скрипт лежит прямо в папке с данными
    data_dir = base_dir

p_noise = {'int_clk': None, 'ext_clk': None}
f_offset = {'int_clk': None, 'ext_clk': None}
f_offset['int_clk'], p_noise['int_clk'] = np.loadtxt(data_dir / 'tr_dat0022.csv', delimiter=',', unpack=True, skiprows=1)
f_offset['ext_clk'], p_noise['ext_clk'] = np.loadtxt(data_dir / 'tr_dat0012.csv', delimiter=',', unpack=True, skiprows=1)

# интерполяция: без сглаживания (raw) и со сглаживанием
f_raw, p_raw, f_interp, p_interp = {}, {}, {}, {}
for name in f_offset:
    print(name)
    f_raw[name], p_raw[name] = interpolate_psd(f_offset[name], p_noise[name], f_min, f_max, df)
    f_interp[name], p_interp[name] = interpolate_psd(f_offset[name], p_noise[name], f_min, f_max, df,
                                                     smooth=SMOOTH, **SMOOTH_KW)

# plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(f_offset['int_clk'], p_noise['int_clk'], 'o', ms=3, mfc='none', c='tab:blue', label='Internal Clock')
ax.plot(f_offset['ext_clk'], p_noise['ext_clk'], 'o', ms=3, mfc='none', c='tab:red', label='External Clock (SRS+Rb Clock)')
ax.plot(f_interp['int_clk'], p_interp['int_clk'], '-', lw=1.0, c='navy', label='Smoothed (Hampel + Savitzky-Golay)')
ax.plot(f_interp['ext_clk'], p_interp['ext_clk'], '-', lw=1.0, c='darkred')

ax.grid(True)
ax.legend(loc='upper right')
ax.set_xscale('log')
ax.set_xlim([f_offset['int_clk'][0], f_offset['int_clk'][-1]])
ax.set_xlabel(r'Frequency Offset, [Hz]')
ax.set_ylabel(r'Phase Noise, [dBc/Hz]')

ax.set_title(r'Advantex Phase Noise (6.834 GHz)')
plt.tight_layout()
# plt.savefig(f'fig_phase_noise.png')

for name, label in [('int_clk', 'Internal'), ('ext_clk', 'External')]:
    ph_raw = count_phase(f_raw[name], p_raw[name])
    ph_sm = count_phase(f_interp[name], p_interp[name])
    print(f"{label} clock: raw   phase = {ph_raw:.4g} rad, gravity = {ph_raw/k_eff/T/T*1e5:.4g} mGal")
    print(f"{label} clock: smooth phase = {ph_sm:.4g} rad, gravity = {ph_sm/k_eff/T/T*1e5:.4g} mGal "
          f"({(ph_sm/ph_raw - 1)*100:+.1f} %)")

plt.show()