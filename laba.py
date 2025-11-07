# === Анализ трибологических испытаний машинного масла с нанотрубками ===
# Назначение: построение графиков зависимостей T(Fn) и M(Fn)
# Формат исходных файлов:
#   0_0.txt - значения силы прижатия Fn [Н]
#   0_1.txt - значения температуры T [°C]
#   0_2.txt - значения момента трения M [Н·м]

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial import Polynomial

# 1. Загрузка данных из txt (необходимо создание либо редактирование получаемых txt файлов)

Fn = np.loadtxt("0_0.txt")
T = np.loadtxt("0_1.txt")
M = np.loadtxt("0_2.txt")

# Проверка соответствия длины массивов
assert len(Fn) == len(T) == len(M), "Количество строк в файлах не совпадает!"
print(f"Загружено {len(Fn)} точек данных.")
print(f"Fn: {Fn.min():.4f}–{Fn.max():.4f} Н | T: {T.min():.4f}–{T.max():.4f} °C | M: {M.min():.4f}–{M.max():.4f} Н·м")

# 2. График 1 — Температура от силы прижатия

p = Polynomial.fit(Fn, T, 2)  #полином 2-го порядка (парабола)
Fn_fit = np.linspace(Fn.min(), Fn.max(), 360)
T_fit = p(Fn_fit)
coeffs = p.convert().coef
r2_T = np.corrcoef(T, p(Fn))[0, 1] ** 2 #коэф аппроксимации

plt.figure(figsize=(7, 5))
plt.scatter(Fn, T,color="tab:blue",s=25,label="Экспериментальные данные") #точки исходных данных
plt.plot(Fn_fit, T_fit,color="tab:orange",lw=2,label=f"Аппроксимация (R²={r2_T:.3f})") #аппроксимация
plt.title("Зависимость температуры от силы прижатия",fontsize=13)
plt.xlabel("Сила прижатия, Н",fontsize=11)
plt.ylabel("Температура, °C",fontsize=11)
plt.grid(True, linestyle="--",alpha=0.6)
plt.legend()
eq_text=f"T={coeffs[2]:.3e}·Fn²+{coeffs[1]:.3f}·Fn+{coeffs[0]:.2f}"
plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.show()

# 3. График 2 — Момент трения от силы прижатия

slope, intercept, r_value, p_value, std_err = stats.linregress(Fn, M)
M_fit = slope * Fn + intercept

plt.figure(figsize=(7, 5))
plt.scatter(Fn, M, color="tab:green", s=25, label="Экспериментальные данные")
plt.plot(Fn, M_fit, color="tab:red", lw=2, label=f"Линейная аппроксимация (R²={r_value**2:.3f})")
plt.title("Зависимость момента трения от силы прижатия", fontsize=13)
plt.xlabel("Сила прижатия, Н", fontsize=11)
plt.ylabel("Момент трения, Н·м", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
eq_text = f"M = {slope:.6f}·Fn + {intercept:.6f}"
plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.show()


# 4. Расчет коэффициента трения μ

# Для оценки μ = M / (Fn * r), если известен радиус контактного плеча r
r=0.01  # [м] — заменить на фактическое значение плеча в мм
mu= M / (Fn * r)

plt.figure(figsize=(7, 5))
plt.plot(Fn, mu, color="tab:purple", lw=1.8)
plt.title("Коэффициент трения в зависимости от силы прижатия", fontsize=13)
plt.xlabel("Сила прижатия, Н", fontsize=11)
plt.ylabel("Коэффициент трения μ", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

print(f"Средний коэффициент трения μ ≈ {mu.mean():.4f}")

# 5. ГРАФИК ВРЕМЕННЫХ ЗАВИСИМОСТЕЙ (добавлено из второго кода)

time = np.arange(len(Fn))
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Сила прижатия во времени
ax1.plot(time, Fn, 'blue', linewidth=1)
ax1.set_ylabel('Сила прижатия, Н', fontsize=12)
ax1.set_title('Динамика силы прижатия во времени', fontsize=13)
ax1.grid(True, alpha=0.3)

# Температура во времени
ax2.plot(time, T, 'red', linewidth=1)
ax2.set_ylabel('Температура, °C', fontsize=12)
ax2.set_title('Динамика температуры во времени', fontsize=13)
ax2.grid(True, alpha=0.3)

# Момент трения во времени
ax3.plot(time, M, 'green', linewidth=1)
ax3.set_ylabel('Момент трения, Н·м', fontsize=12)
ax3.set_xlabel('Номер измерения', fontsize=12)
ax3.set_title('Динамика момента трения во времени', fontsize=13)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()