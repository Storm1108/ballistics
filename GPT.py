import numpy as np
import pandas as pd
from datetime import datetime
from nrlmsise00 import msise_model
from math import sin, cos, radians, sqrt
from matplotlib import pyplot as plt
import logging
from numba import jit

# Математическая модель движения
@jit
def right_sides(t, state_vec, rho):
    derives = np.zeros(6, dtype=np.float64)
    r_abs_m = np.linalg.norm(state_vec[0:3])
    speed = np.linalg.norm(state_vec[3:6])
    derives[0:3] = state_vec[3:6]
    derives[3:6] = -state_vec[0:3] * mu_m3s2/r_abs_m**3 - sigma * rho * speed * state_vec[3:6]
    return derives

# Численный метод Рунге-Кутты
@jit
def RK4step(t1, state_vec1, rho, h_s=0.5):
    k1 = right_sides(t1, state_vec1, rho)
    k2 = right_sides(t1 + 0.5 * h_s, state_vec1 + 0.5 * h_s * k1, rho)
    k3 = right_sides(t1 + 0.5 * h_s, state_vec1 + 0.5 * h_s * k2, rho)
    k4 = right_sides(t1 + h_s, state_vec1 + h_s * k3, rho)
    state_vec2 = state_vec1 + h_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0
    t2 = t1 + h_s
    return t2, state_vec2

# Исходные условия и параметры
mu_m3s2 = 3.986004419e+14  # Гравитационный параметр Земли
re_m = 6371e+03            # Радиус Земли
h0_m = 285e+03             # Начальная высота
r = re_m + h0_m            # Начальный радиус
V0_ms = sqrt(mu_m3s2/r)    # Начальная скорость
i0_deg = 22                # Наклонение орбиты
Vy0 = V0_ms*cos(radians(i0_deg))  # Компонента начальной скорости по Y
Vz0 = V0_ms*sin(radians(i0_deg))  # Компонента начальной скорости по Z
sigma = 0.05               # Баллистический коэффициент

state_vec0 = [r, 0.0, 0.0, 0.0, Vy0, Vz0]

# Расчет траектории
t0 = 0.0
t1 = t0
state_vec1 = np.array(state_vec0, dtype=np.float64)

# Список для хранения результатов
results = []

# Расчет каждой точки траектории
while (np.linalg.norm(state_vec1[0:3])-re_m) > 100e+03:
    h_km = (np.linalg.norm(state_vec1[0:3]) / 1000) - (re_m / 1000)
    F107 = 125
    F81 = F107
    a = msise_model(datetime(2024, 1, 1, 0, 0, 0), h_km, 0, 0, F107, F81, 4, lst=16)
    rho = a[0][5]*1000  # функция плотности по MSISE
    t2, state_vec2 = RK4step(t1, state_vec1, rho)
    # Логирование для отладки
    logging.info(f'Текущее время: {t2}, Позиция: {state_vec2[0:3]}, Скорость: {state_vec2[3:]}')
    # Добавляем результаты в список
    current_height = (np.linalg.norm(state_vec2[0:3]) - re_m) / 1000  # Высота в километрах
    results.append([t2, state_vec2[0], state_vec2[1], state_vec2[2], state_vec2[3], state_vec2[4], state_vec2[5], current_height])
    t1 = t2
    state_vec1 = state_vec2

# Создаем DataFrame из списка результатов
results_df = pd.DataFrame(results, columns=['t', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'height'])

# Сохраняем результаты в Excel
results_df.to_excel('output.xlsx', index=False)

# Подготовка к отображению обоих графиков
plt.figure(figsize=(14, 7))

# График высоты от времени
plt.subplot(1, 2, 1)  # 1 строка, 2 колонки, первый график
plt.plot(results_df['t'], results_df['height'], label='Высота от времени')  # Время уже в часах, высота в км
plt.xlabel('Время (ч)')
plt.ylabel('Высота (км)')
plt.title('График высоты от времени')
plt.legend()
plt.grid(True)

# Визуализация траектории
plt.subplot(1, 2, 2, projection='3d')  # 1 строка, 2 колонки, второй график
plt.plot(results_df['x'], results_df['y'], results_df['z'])
plt.title('Траектория')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.grid(True)

# Показать оба графика
plt.tight_layout()
plt.show()
