import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def plot_s_params_comparison_separate(freq, S11_raw, S21_raw, S11_deemb, S21_deemb, title_prefix):
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig1.suptitle(f'{title_prefix} - Амплитуды S-параметров', fontsize=14, fontweight='bold')
    ax1.plot(freq / 1e9, 20 * np.log10(np.abs(S11_raw)), 'b-',
             label='До деэмбеддинга', linewidth=2, alpha=0.7)
    ax1.plot(freq / 1e9, 20 * np.log10(np.abs(S11_deemb)), 'r--',
             label='После деэмбеддинга', linewidth=2, alpha=0.9)
    ax1.set_xlabel('Частота (ГГц)')
    ax1.set_ylabel('|S11| (дБ)')
    ax1.set_title('Амплитуда S11')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(freq / 1e9, 20 * np.log10(np.abs(S21_raw)), 'g-',
             label='До деэмбеддинга', linewidth=2, alpha=0.7)
    ax2.plot(freq / 1e9, 20 * np.log10(np.abs(S21_deemb)), 'm--',
             label='После деэмбеддинга', linewidth=2, alpha=0.9)
    ax2.set_xlabel('Частота (ГГц)')
    ax2.set_ylabel('|S21| (дБ)')
    ax2.set_title('Амплитуда S21')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    fig2.suptitle(f'{title_prefix} - Фазы S-параметров', fontsize=14, fontweight='bold')
    ax3.plot(freq / 1e9, np.angle(S11_raw, deg=True), 'b-',
             label='До деэмбеддинга', linewidth=2, alpha=0.7)
    ax3.plot(freq / 1e9, np.angle(S11_deemb, deg=True), 'r--',
             label='После деэмбеддинга', linewidth=2, alpha=0.9)
    ax3.set_xlabel('Частота (ГГц)')
    ax3.set_ylabel('∠S11 (градусы)')
    ax3.set_title('Фаза S11')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax4.plot(freq / 1e9, np.angle(S21_raw, deg=True), 'g-',
             label='До деэмбеддинга', linewidth=2, alpha=0.7)
    ax4.plot(freq / 1e9, np.angle(S21_deemb, deg=True), 'm--',
             label='После деэмбеддинга', linewidth=2, alpha=0.9)
    ax4.set_xlabel('Частота (ГГц)')
    ax4.set_ylabel('∠S21 (градусы)')
    ax4.set_title('Фаза S21')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_all_s_params__separate(freq, S11, S21, S12, S22, title_prefix):
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f'{title_prefix} - Амплитуды исходных S-параметров', fontsize=14, fontweight='bold')
    ax1.plot(freq / 1e9, 20 * np.log10(np.abs(S11)), 'b-', label='|S11|', linewidth=2)
    ax1.set_xlabel('Частота (ГГц)')
    ax1.set_ylabel('|S11| (дБ)')
    ax1.set_title('Амплитуда S11')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(freq / 1e9, 20 * np.log10(np.abs(S21)), 'r-', label='|S21|', linewidth=2)
    ax2.set_xlabel('Частота (ГГц)')
    ax2.set_ylabel('|S21| (дБ)')
    ax2.set_title('Амплитуда S21')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3.plot(freq / 1e9, 20 * np.log10(np.abs(S12)), 'g-', label='|S12|', linewidth=2)
    ax3.set_xlabel('Частота (ГГц)')
    ax3.set_ylabel('|S12| (дБ)')
    ax3.set_title('Амплитуда S12')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax4.plot(freq / 1e9, 20 * np.log10(np.abs(S22)), 'm-', label='|S22|', linewidth=2)
    ax4.set_xlabel('Частота (ГГц)')
    ax4.set_ylabel('|S22| (дБ)')
    ax4.set_title('Амплитуда S22')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f'{title_prefix} - Фазы исходных S-параметров', fontsize=14, fontweight='bold')
    ax5.plot(freq / 1e9, np.angle(S11, deg=True), 'b--', label='∠S11', linewidth=2)
    ax5.set_xlabel('Частота (ГГц)')
    ax5.set_ylabel('∠S11 (градусы)')
    ax5.set_title('Фаза S11')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax6.plot(freq / 1e9, np.angle(S21, deg=True), 'r--', label='∠S21', linewidth=2)
    ax6.set_xlabel('Частота (ГГц)')
    ax6.set_ylabel('∠S21 (градусы)')
    ax6.set_title('Фаза S21')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax7.plot(freq / 1e9, np.angle(S12, deg=True), 'g--', label='∠S12', linewidth=2)
    ax7.set_xlabel('Частота (ГГц)')
    ax7.set_ylabel('∠S12 (градусы)')
    ax7.set_title('Фаза S12')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax8.plot(freq / 1e9, np.angle(S22, deg=True), 'm--', label='∠S22', linewidth=2)
    ax8.set_xlabel('Частота (ГГц)')
    ax8.set_ylabel('∠S22 (градусы)')
    ax8.set_title('Фаза S22')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def read_s2p_file(filename):
    freq = []
    s11_mag, s11_deg = [], []
    s21_mag, s21_deg = [], []
    s12_mag, s12_deg = [], []
    s22_mag, s22_deg = [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
    reading_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('!'):
            continue
        if line.startswith('#'):
            reading_data = True
            continue
        if reading_data:
            parts = line.split()
            if len(parts) >= 9:
                freq.append(float(parts[0]))
                s11_mag.append(float(parts[1]))
                s11_deg.append(float(parts[2]))
                s21_mag.append(float(parts[3]))
                s21_deg.append(float(parts[4]))
                s12_mag.append(float(parts[5]))
                s12_deg.append(float(parts[6]))
                s22_mag.append(float(parts[7]))
                s22_deg.append(float(parts[8]))
    freq = np.array(freq)
    s11_complex = 10 ** (np.array(s11_mag) / 20) * np.exp(1j * np.deg2rad(s11_deg))
    s21_complex = 10 ** (np.array(s21_mag) / 20) * np.exp(1j * np.deg2rad(s21_deg))
    s12_complex = 10 ** (np.array(s12_mag) / 20) * np.exp(1j * np.deg2rad(s12_deg))
    s22_complex = 10 ** (np.array(s22_mag) / 20) * np.exp(1j * np.deg2rad(s22_deg))
    return freq, s11_complex, s21_complex, s12_complex, s22_complex

def apply_waveguide_deembedding(S11, S21, S12, S22, freq_Hz, L1, L2, fc):
    c = 299792458.0
    beta = np.zeros_like(freq_Hz, dtype=complex)
    mask = freq_Hz > fc
    beta[mask] = (2 * np.pi * freq_Hz[mask] / c) * np.sqrt(1 - (fc / freq_Hz[mask]) ** 2)
    R1 = np.exp(-1j * beta * L1)
    R2 = np.exp(-1j * beta * L2)
    S11_corr = S11 / (R1 * R1)
    S22_corr = S22 / (R2 * R2)
    S21_corr = S21 / (R1 * R2)
    S12_corr = S12 / (R1 * R2)
    return S11_corr, S21_corr, S12_corr, S22_corr

def stepwise_nrw_method(S11, S21, freq_Hz, d, fc):
    c = 299792458.0
    k0 = 2 * np.pi * freq_Hz / c
    omega = 2 * np.pi * freq_Hz
    n_points = len(freq_Hz)
    eps_r = np.full(n_points, complex(np.nan, np.nan))
    mu_r = np.full(n_points, complex(np.nan, np.nan))
    z = np.full(n_points, complex(np.nan, np.nan))
    numerator = (1 + S11) ** 2 - S21 ** 2
    denominator = (1 - S11) ** 2 - S21 ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.sqrt(numerator / denominator)
    z = np.where(z.real >= 0, z, -z)
    with np.errstate(invalid='ignore', divide='ignore'):
        term1 = (1 - S11 ** 2 + S21 ** 2) / (2 * S21)
        z_term = z - 1 / z
        term2 = (2 * S11) / (z_term * S21)
    exp_gamma_d = term1 + term2
    A = np.abs(exp_gamma_d)
    phi_raw = np.angle(exp_gamma_d)
    lambda_min = c / freq_Hz[0]
    if d < lambda_min / 2:
        m0 = 0
    else:
        beta_est = k0[0] * np.sqrt(3 * 1)
        m0 = int(np.round((beta_est * d) / (2 * np.pi)))
        m0 = max(m0, 0)
    phi_corrected = np.zeros_like(phi_raw)
    phi_corrected[0] = phi_raw[0] + 2 * np.pi * m0
    for i in range(1, n_points):
        delta_phi = phi_raw[i] - phi_raw[i - 1]
        if np.abs(delta_phi) >= np.pi:
            if i >= 2:
                phi_corrected[i] = 2 * phi_corrected[i - 1] - phi_corrected[i - 2]
            else:
                phi_corrected[i] = phi_corrected[i - 1]
        else:
            phi_corrected[i] = phi_corrected[i - 1] + delta_phi
    omega_c = 2 * np.pi * fc
    sqrt_term = (-1j * np.log(A) + phi_corrected) / (k0 * d)
    n_sq = (sqrt_term) ** 2 + (omega_c / omega) ** 2
    root_numer = n_sq - (fc / freq_Hz) ** 2
    root_denom = 1 - (fc / freq_Hz) ** 2
    root_term = np.sqrt(root_numer / root_denom)
    mu_r = z * root_term
    eps_r = n_sq / mu_r
    eps_real_valid = eps_r.real
    freq_valid = freq_Hz
    x = freq_valid
    y = eps_real_valid
    f_interp = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    eps_real_valid = f_interp(freq_valid)
    eps_r.real = eps_real_valid
    T_corrected = A * np.exp(1j * phi_corrected)
    return eps_r, mu_r, z, T_corrected

def calculate_for_frequency_range(filename, sample_length_m, waveguide_width_m):
    freq, S11, S21, S12, S22 = read_s2p_file(filename)
    plot_all_s_params__separate(freq, S11, S21, S12, S22, "Исходные данные")
    freq_min_ghz = 8
    freq_max_ghz = 12
    freq_selected = freq
    S11_selected = S11
    S21_selected = S21
    S12_selected = S12
    S22_selected = S22
    c = 299792458.0
    a = waveguide_width_m
    fc = c / (2 * a)
    print(f"Критическая частота волновода: {fc / 1e9:.3f} ГГц")
    total_wg_length = 0.54
    L = (total_wg_length - sample_length_m) / 2.0
    print(f"Расстояние до образца: L = {L * 1000:.1f} мм")
    S11_corr, S21_corr, S12_corr, S22_corr = apply_waveguide_deembedding(
        S11_selected, S21_selected, S12_selected, S22_selected,
        freq_selected, L, L, fc
    )

    S11_avg = (S11_corr + S22_corr) / 2
    S21_avg = (S21_corr + S12_corr) / 2
    print("\nСравнение S-параметров до и после деэмбеддинга...")
    plot_s_params_comparison_separate(freq_selected, S11_selected, S21_selected,
                                      S11_avg, S21_avg,
                                      f"{freq_min_ghz:.1f}-{freq_max_ghz:.1f} ГГц")
    eps_r, mu_r, z, T_corrected = stepwise_nrw_method(
        S11_avg, S21_avg, freq_selected, sample_length_m, fc
    )

    eps_real = eps_r.real
    eps_imag = -eps_r.imag
    mu_real = mu_r.real
    mu_imag = -mu_r.imag
    print(f"Диэлектрическая проницаемость в выбранном диапазоне:")
    print(f"  ε' : среднее = {np.mean(eps_real):.4f}, min = {np.min(eps_real):.4f}, max = {np.max(eps_real):.4f}")
    print(f"  ε'' : среднее = {np.mean(eps_imag):.4f}, min = {np.min(eps_imag):.4f}, max = {np.max(eps_imag):.4f}")
    print(f"  tgδ = ε''/ε' : среднее = {np.mean(eps_imag / eps_real):.6f}")
    print(f"\nМагнитная проницаемость в выбранном диапазоне:")
    print(f"  µ' : среднее = {np.mean(mu_real):.4f}, min = {np.min(mu_real):.4f}, max = {np.max(mu_real):.4f}")
    print(f"  µ'' : среднее = {np.mean(mu_imag):.4f}, min = {np.min(mu_imag):.4f}, max = {np.max(mu_imag):.4f}")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    ax1 = axes[0, 0]
    ax1.plot(freq_selected / 1e9, 20 * np.log10(np.abs(S11_avg)), 'b-', label='|S11|', alpha=0.7, linewidth=1.5)
    ax1.plot(freq_selected / 1e9, 20 * np.log10(np.abs(S21_avg)), 'r-', label='|S21|', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Частота (ГГц)', fontsize=11)
    ax1.set_ylabel('Амплитуда (дБ)', fontsize=11)
    ax1.set_title(f'Амплитуды S-параметров после усреднения', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2 = axes[0, 1]
    ax2.plot(freq_selected / 1e9, np.angle(S11_avg, deg=True), 'b--', label='∠S11', alpha=0.5, linewidth=1)
    ax2.plot(freq_selected / 1e9, np.angle(S21_avg, deg=True), 'r--', label='∠S21', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Частота (ГГц)', fontsize=11)
    ax2.set_ylabel('Фаза (градусы)', fontsize=11)
    ax2.set_title(f'Фазы S-параметров после усреднения', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax3 = axes[1, 0]
    ax3.plot(freq_selected / 1e9, eps_r.real, 'b-',
             label="ε'", linewidth=2, alpha=0.8, marker='o', markersize=4)
    mean_eps = np.mean(eps_r.real)
    ax3.axhline(y=mean_eps, color='red', linestyle=':', alpha=0.7,
                label=f"Среднее = {mean_eps:.3f}")

    ax3.set_xlabel('Частота (ГГц)', fontsize=11)
    ax3.set_ylabel("ε'", fontsize=11)
    ax3.set_title(f"Действительная часть ε",
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax4 = axes[1, 1]
    ax4.plot(freq_selected / 1e9, -eps_r.imag, 'r-',
             label="ε''", linewidth=2, alpha=0.8, marker='s', markersize=4)
    mean_eps_imag = np.mean(-eps_r.imag)
    ax4.axhline(y=mean_eps_imag, color='blue', linestyle=':', alpha=0.7,
                label=f"Среднее = {mean_eps_imag:.4f}")

    ax4.set_xlabel('Частота (ГГц)', fontsize=11)
    ax4.set_ylabel("ε''", fontsize=11)
    ax4.set_title(f"Мнимая часть ε",
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax5 = axes[2, 0]
    ax5.plot(freq_selected / 1e9, mu_r.real, 'g-',
             label="µ'", linewidth=2, alpha=0.8, marker='^', markersize=4)
    mean_mu = np.mean(mu_r.real)
    ax5.axhline(y=mean_mu, color='red', linestyle=':', alpha=0.7,
                label=f"Среднее = {mean_mu:.3f}")
    ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.5,
                label='µ=1 (ожидание)')

    ax5.set_xlabel('Частота (ГГц)', fontsize=11)
    ax5.set_ylabel("µ'", fontsize=11)
    ax5.set_title(f"Действительная часть µ",
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10, loc='best')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.tick_params(axis='both', which='major', labelsize=10)
    ax6 = axes[2, 1]

    ax6.plot(freq_selected / 1e9, -mu_r.imag, 'm-',
             label="µ''", linewidth=2, alpha=0.8, marker='v', markersize=4)
    mean_mu_imag = np.mean(-mu_r.imag)
    ax6.axhline(y=mean_mu_imag, color='blue', linestyle=':', alpha=0.7,
                label=f"Среднее = {mean_mu_imag:.4f}")
    ax6.set_xlabel('Частота (ГГц)', fontsize=11)
    ax6.set_ylabel("µ''", fontsize=11)
    ax6.set_title(f"Мнимая часть µ",
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10, loc='best')
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()
    return {
        'frequency': freq_selected,
        'eps_r': eps_r,
        'mu_r': mu_r,
        'S11_avg': S11_avg,
        'S21_avg': S21_avg,
        'T_corrected': T_corrected,
        'fc': fc,
        'freq_range_ghz': (freq_min_ghz, freq_max_ghz)
    }

if __name__ == "__main__":
    print("=" * 70)
    print("РАСЧЕТ ПАРАМЕТРОВ МАТЕРИАЛА С ВЫБОРОМ ДИАПАЗОНА ЧАСТОТ")
    print("=" * 70)
    filename = input("Введите имя файла s2p [по умолчанию: measurements.s2p]: ") or "measur.s2p"
    sample_length_mm = float(input("Введите длину образца (мм) [по умолчанию: 30.0]: ") or "30.0")
    waveguide_width_mm = float(input("Введите ширину волновода (мм) [по умолчанию: 22.86]: ") or "22.86")
    sample_length_m = sample_length_mm / 1000.0
    waveguide_width_m = waveguide_width_mm / 1000.0
    print(f"\nПараметры расчета:")
    print(f"  Файл: {filename}")
    print(f"  Длина образца: {sample_length_mm} мм")
    print(f"  Ширина волновода: {waveguide_width_mm} мм")
    try:
        results = calculate_for_frequency_range(
            filename=filename,
            sample_length_m=sample_length_m,
            waveguide_width_m=waveguide_width_m
        )
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден.")
    except Exception as e:
        print(f"Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()