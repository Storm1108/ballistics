import numpy as np, math as m, time
from matplotlib import pyplot as plt
from matplotlib import use

use('TkAgg')
import pandas as pd
from datetime import datetime
from nrlmsise00 import msise_model


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

def RK(p, dt, h_lim, mu_z, sigm, R_km)
    k1 = f(p, mu_z, sigm, R_km)  # коэффициенты Рунге-Кутта
    k2 = f(p + dt * k1 / 2, mu_z, sigm, R_km)
    k3 = f(p + dt * k2 / 2, mu_z, sigm, R_km)
    k4 = f(p + dt * k3, mu_z, sigm, R_km)
    p = p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # формула Рунге-Кутты
    return
def f(p, mu_z, sigm, R_km):  # функция правых частей уравнений
    f = np.zeros([6])
    f[0:3] = p[3:6]  # dx/dt=vx
    r_m = float(np.linalg.norm(p[0:3]))
    v_m = float(np.linalg.norm(p[3:6]))
    f[3] = (-mu_z * p[0] / (r_m) ** 3
            - sigm * density(r_m / 1000 - R_km) * v_m * p[3])  # dvx/dt=-mu_z*x/r^3-sigma*rho*v*vx
    f[4] = (-mu_z * p[1] / (r_m) ** 3
            - sigm * density(r_m / 1000 - R_km) * v_m * p[4])  # dvy/dt=-mu_z*y/r^3-sigma*rho*v*vy
    f[5] = (-mu_z * p[2] / (r_m) ** 3
            - sigm * density(r_m / 1000 - R_km) * v_m * p[5])  # dvz/dt=-mu_z*z/r^3-sigma*rho*v*vz
    return f


def RK4(f, p0, dt, h_lim, mu_z, sigm, R_km):  # метод Рунге-Кутта 4 порядка
    ti = 0  # текущее время внутри интеграла
    count = 0
    times = [ti]  # список времени
    states = []
    states.append(p0)
    p = p0  # задаем буферный список
    start_t = time.time()  # начинаем отсчет действительного времени интегрирования
    while np.linalg.norm(p[0:3]) / 1000 - R_km > h_lim:  # условие выхода из цикла
        RK(p)
        ti += dt
        count += 1
        p2=p
        if count >= 1000:
            states.append(tuple(p))
            times.append(ti)
            count = 0
        if ti % 36000 == 0:
                print("Текущее время: ", ti / 3600, "ч;",
                      "координата x: ", p[0], "м;",
                      "координата y: ", p[1], "м;",
                      "координата z: ", p[2], "м;",
                      "скорость Vx: ", p[3], "м/c;",
                      "скорость Vy: ", p[4], "м/c;",
                      "скорость Vz: ", p[5], "м/c;",
                      "Время интегрирования: ", time.time() - start_t, "c;")  # выводим параметры каждые 10 часов

    return states, times


def h_to_km(R_km):  # Создание массива высот КА с размерностью в км
    # Задание списка высот в км
    heightinkm = []
    # Запись всех элементов результата интегрирования в новый список и перевод в размерность км
    for i in range(0, len(states_res)):
        heightinkm.append(float(np.linalg.norm(states_res[i][0:3])) / 1000 - R_km)
    return heightinkm


## Начальные условия и константы
mu = 398600.45e+9  # гравитационный параметр
R_km = 6371  # радиус Земли в км
h_km = 220  # начальная высота в км
incl_rad = m.radians(30)  # наклонение орбиты
F107 = 140  # иднекс F107
sigma = 0.0015  # баллистический коэффициент
r = (R_km + h_km) * 10 ** 3  # начальный радиус-вектор
V0 = m.sqrt(mu / r)  # начальная скорость КА
x0 = r  # начальные координаты
y0 = 0
z0 = 0
Vx0 = 0  # начальные скорости
Vy0 = V0 * m.cos(incl_rad)
Vz0 = V0 * m.sin(incl_rad)
dt = 100  # шаг интегрирования
h_lim = 100  # предельная высота интегрирования
p0 = np.array([x0, y0, z0, Vx0, Vy0, Vz0])  # массив numpy с начальными параметрами
start_t = time.time()  # время начала отсчета интегрирования
states_res, times_res = RK4(f, p0, dt, h_lim, mu, sigma, R_km)  # вызов функции Рунге-Кутта
states_math = np.array(states_res)
print("Интегрирование успешно")
print("Всего интегрирование заняло:", time.time() - start_t, "c")
height = h_to_km(R_km)

plt.figure(1, figsize=(9, 3))
plt.subplot(111)
plt.title("Высота")  # Построение графика h от t
plt.xlabel("t, сек")
plt.ylabel("h, км")
plt.plot(times_res, height)
resultplot = plt.figure().add_subplot(projection="3d", elev=45, azim=135)
resultplot.set_title("Траектория")  # Построение 3d траектории
resultplot.set_xlabel("x, км")
resultplot.set_ylabel("y, км")
resultplot.set_zlabel("z, км")
sf_u, sf_v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
# setting x, y, z co-ordinates
sf_x = 6371e+3 * np.cos(sf_u) * np.sin(sf_v)
sf_y = 6371e+3 * np.sin(sf_u) * np.sin(sf_v)
sf_z = 6371e+3 * np.cos(sf_v)

# plotting the curve
resultplot.plot(states_math[:,0], states_math[:,1], states_math[:,2], color='red', visible=True, mouseover=True, zorder=1)
resultplot.plot_surface(sf_x, sf_y, sf_z, rstride=5, cstride=5, linewidth=1, zorder=0, alpha=0.3)

plt.show()
