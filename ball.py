from datetime import datetime
import pandas as pd
import numpy as np
from math import sin, cos, radians, sqrt
from matplotlib import pyplot as plt
from matplotlib import use
from nrlmsise00 import msise_model


use('TkAgg')


# МАТМОДЕЛЬ ДВИЖЕНИЯ
def density(H_km):
    F107 = 100
    F81 = F107
    """F0 = 150
    rho0 = 1.58868 * 10 ** -8 # плотность ночной атмосферы на 120 км
    if H_km > 120:
        a0 = 29.6418  # коэф-ы для расчета плотности ночной атмосферы
        a1 = -0.514957
        a2 = 0.00341926
        a3 = -1.25785 * 10 ** -5
        a4 = 2.5727 * 10 ** -8                          #функция плотности по ГОСТ Р 25645.166-2004
        a5 = -2.75874 * 10 ** -11
        a6 = 1.21091 * 10 ** -14
        l0 = -1.31444
        l1 = 0.0133124
        l2 = -2.55585 * 10 ** -5
        l3 = 5.43981 * 10 ** -8
        l4 = -4.33784 * 10 ** -11
        rhon = rho0 * m.exp(a0 + a1*H_km + a2*H_km**2 + a3*H_km**3 + a4*H_km**4 + a5*H_km**5 + a6*H_km**6)
        K0 = 1 + (l0 + l1*H_km + l2*H_km**2 + l3*H_km**3 + l4*H_km**4)*(F81-F0)/F0
        rho = rhon * K0
    else:
        a04 = 3.66 * 10 ** -7
        k14 = -0.18553
        k24 = 1.5397 * 10 ** -3
        rho = a04 * m.exp(k14 * (H_km - 100) + k24 * (H_km - 100) ** 2)"""
    a = msise_model(datetime(2024, 1, 1, 0, 0, 0), H_km, 0, 0, F107, F81, 4, lst=16)
    rho = a[0][5] * 1000  # функция плотности по MSISE
    return rho

def right_sides(t, state_vec):
    # state_vec = [x, y, z, V_x, V_y, V_z]
    derivs = np.array([0.0]*6, dtype=np.float64)    # d/dt[x, y, z, V_x, V_y, V_z]
    r_m = float(np.linalg.norm(state_vec[0:3]))
    v_m = float(np.linalg.norm(state_vec[3:6]))
    derivs[0:3] = state_vec[3:6]
    derivs[3] = (-mu_m3s2 * state_vec[0] / (r_m) ** 3
            - sigma * density(r_m / 1000 - R_km) * v_m * state_vec[3])  # dvx/dt=-mu_z*x/r^3-sigma*rho*v*vx
    derivs[4] = (-mu_m3s2 * state_vec[1] / (r_m) ** 3
            - sigma * density(r_m / 1000 - R_km) * v_m * state_vec[4])  # dvy/dt=-mu_z*y/r^3-sigma*rho*v*vy
    derivs[5] = (-mu_m3s2 * state_vec[2] / (r_m) ** 3
            - sigma * density(r_m / 1000 - R_km) * v_m * state_vec[5])  # dvz/dt=-mu_z*z/r^3-sigma*rho*v*vz
    return derivs

# ЧИСЛЕННЫЙ МЕТОД РУНГЕ-КУТТЫ

def RK4step(t1, state_vec1):
    h_s = 10

    k1 = right_sides(t1, state_vec1)
    k2 = right_sides(t1 + 0.5 * h_s, state_vec1 + 0.5 * h_s * k1)
    k3 = right_sides(t1 + 0.5 * h_s, state_vec1 + 0.5 * h_s * k2)
    k4 = right_sides(t1 + h_s, state_vec1 + h_s * k3)

    state_vec2 = state_vec1 + h_s * (k1 + 2.0*k2 +2.0*k3 + k4) / 6.0
    t2 = t1 + h_s

    return [t2, state_vec2]

# РАСЧЁТ ТРАЕКТОРИИ ОРБИТЫ
sigma = 0.0015
mu_m3s2 = 3.986004419e+14
re_m = 6371e+03
R_km = 6371
h0_m = 200e+03
V0_ms = sqrt(mu_m3s2/(re_m+h0_m))
i0_deg = 45.0
Vy0 = V0_ms*cos(radians(i0_deg))
Vz0 = V0_ms*sin(radians(i0_deg))
state_vec0 = [
        re_m+h0_m,  0.0,        0.0,
        0.0,        Vy0,        Vz0
]

t0 = 0.0
t1 = t0
state_vec1 = np.array(state_vec0, dtype=np.float64)

times = [t1, ]
states = [state_vec1/1000, ]
heights = [h0_m/1000, ]

print_counter = 0
while float(np.linalg.norm(state_vec1[0:3]))-re_m > 100*10**3:
    t2, state_vec2 = RK4step(t1, state_vec1)

    if print_counter == 10:
        times.append(t2/86400)
        states.append(state_vec2/1000)
        heights.append(float(np.linalg.norm(state_vec2[0:3]))/1000-R_km)
        print_counter = 0

    t1 = t2
    state_vec1 = state_vec2
    print_counter += 1

states_math = np.array(states)
results = np.column_stack([times, states, heights])
print(results[:10])

# Создаем DataFrame из списка результатов
results_df = pd.DataFrame(results, columns=['t', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'height'])

# Сохраняем результаты в Excel
results_df.to_excel('output.xlsx', index=False)

plt.subplot(1, 2, 1)  # 1 строка, 2 колонки, первый график
plt.plot(results_df['t'], results_df['height'], label='Высота от времени')  # Время уже в часах, высота в км
plt.xlabel('Время (ч)')
plt.ylabel('Высота (км)')
plt.title('График высоты от времени')
plt.legend()
plt.grid(True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(states_math[:, 0], states_math[:, 1], states_math[:, 2])
sf_u, sf_v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
# setting x, y, z co-ordinates
sf_x = 6371 * np.cos(sf_u) * np.sin(sf_v)
sf_y = 6371 * np.sin(sf_u) * np.sin(sf_v)
sf_z = 6371 * np.cos(sf_v)
ax.plot_surface(sf_x, sf_y, sf_z, rstride=5, cstride=5, linewidth=1, zorder=0, alpha=0.3)
plt.grid()
plt.show()