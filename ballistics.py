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
    a = msise_model(datetime(2020, 1, 1, 0, 0, 0), H_km, 0, 0, F107, F81, 4, lst=16)
    rho = a[0][5] * 1000  # функция плотности по MSISE
    return rho


def f(p, mu_z, sigm, R_km):  # функция правых частей уравнений
    f = np.zeros([8])
    f[0] = p[3]  # dx/dt=vx
    f[1] = p[4]  # dy/dt=vy
    f[2] = p[5]  # dz/dt=vz
    f[3] = (-mu_z * p[0] / (p[6]) ** 3
            - sigm * density(p[6] / 1000 - R_km) * p[7] * p[3])  # dvx/dt=-mu_z*x/r^3-sigma*rho*v*vx
    f[4] = (-mu_z * p[1] / (p[6]) ** 3
            - sigm * density(p[6] / 1000 - R_km) * p[7] * p[4])  # dvy/dt=-mu_z*y/r^3-sigma*rho*v*vy
    f[5] = (-mu_z * p[2] / (p[6]) ** 3
            - sigm * density(p[6] / 1000 - R_km) * p[7] * p[5])  # dvz/dt=-mu_z*z/r^3-sigma*rho*v*vz
    return f


def RK4(f, p0, dt, h_lim, mu_z, sigm, R_km):  # метод Рунге-Кутта 4 порядка
    ti = 0  # текущее время внутри интеграла
    x_list = []  # списки с координатами
    y_list = []
    z_list = []
    Vx_list = []  # списки со скоростями
    Vy_list = []
    Vz_list = []
    h_list = []  # список высот
    t_list = []  # список времени
    res_p = [x_list, y_list, z_list, Vx_list, Vy_list, Vz_list, h_list, t_list]
    for i in range(7):
        res_p[i].append(p0[i])  # добавляем начальные условия
    res_p[7].append(ti)  # и время
    p = p0  # задаем буферный список
    start_t = time.time()  # начинаем отсчет действительного времени интегрирования
    while p[6] / 1000 - R_km > h_lim:  # условие выхода из цикла
        k1 = f(p, mu_z, sigm, R_km)  # коэффициенты Рунге-Кутта
        k2 = f(p + dt * k1 / 2, mu_z, sigm, R_km)
        k3 = f(p + dt * k2 / 2, mu_z, sigm, R_km)
        k4 = f(p + dt * k3, mu_z, sigm, R_km)
        p = p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # формула Рунге-Кутты
        p[6] = ((p[0]) ** 2 + (p[1]) ** 2 + (p[2]) ** 2) ** 0.5  # вычисляем текущий модуль радиуса-вектора
        p[7] = ((p[3]) ** 2 + (p[4]) ** 2 + (p[5]) ** 2) ** 0.5  # и скорость
        ti += dt
        for i in range(7):
            res_p[i].append(p[i])  # добавляем значения в результат интегрирования
        res_p[7].append(ti)
        if ti % 36000 == 0:
            print("Текущее время: ", ti / 3600, "ч;",
                  "координата x: ", p[0], "м;",
                  "координата y: ", p[1], "м;",
                  "координата z: ", p[2], "м;",
                  "скорость Vx: ", p[3], "м/c;",
                  "скорость Vy: ", p[4], "м/c;",
                  "скорость Vz: ", p[5], "м/c;",
                  "высота орбиты: ", p[6] / 1000 - R_km, "км;",
                  "Время интегрирования: ", time.time() - start_t, "c;")  # выводим параметры каждые 10 часов
    return res_p


def sort(res_p, cnt, xsort, ysort, zsort, vxsort, vysort, vzsort, tsort):
    # Сортировка полученных данных для вывода в файл .xlsx
    xsort.extend(res_p[0])  # копируем данные
    ysort.extend(res_p[1])
    zsort.extend(res_p[2])
    vxsort.extend(res_p[3])
    vysort.extend(res_p[4])
    vzsort.extend(res_p[5])
    tsort.extend(res_p[7])

    for j in [xsort, ysort, zsort, vxsort, vysort, vzsort, tsort]:
        ilist = []  # список индексов
        asort = []  # список с элементами под этими индексами
        k = len(j) // (cnt - 1)  # вычисление шага, каждый к-тый элемент будет оставляться
        # прохождение массива от начала до предпоследнего элемента, так как последний элемент оставляем
        for i in range(len(j) - 2):
            # Индексы элементов добавляются в список, если индекс кратнен k
            if i % k == 0:
                ilist.append(i)
        for i in ilist:  # в список добавляются элементы с индексами, кратными k
            asort.append(j[i])
        if ilist[-1] != len(j) - 1:  # Если последний индекс списка не кратен k, то последний элемент
            asort.append(j[-1])  # все равно добавляется в список, чтобы их число равнялось 20
        j.clear()
        j.extend(asort)


def h_to_km(R_km):  # Создание массива высот КА с размерностью в км
    # Задание списка высот в км
    heightinkm = []
    # Запись всех элементов результата интегрирования в новый список и перевод в размерность км
    for i in range(0, len(res_p[6])):
        heightinkm.append(res_p[6][i] / 1000 - R_km)
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
x0 = r * m.cos(incl_rad)  # начальные координаты
y0 = 0
z0 = r * m.sin(incl_rad)
Vx0 = 0  # начальные скорости
Vy0 = V0
Vz0 = 0
dt = 100  # шаг интегрирования
h_lim = 100  # предельная высота интегрирования
p0 = np.array([x0, y0, z0, Vx0, Vy0, Vz0, 0, 0])  # массив numpy с начальными параметрами
p0[6] = ((p0[0]) ** 2 + (p0[1]) ** 2 + (p0[2]) ** 2) ** 0.5  # начальный радиус-вектор
p0[7] = ((p0[3]) ** 2 + (p0[4]) ** 2 + (p0[5]) ** 2) ** 0.5  # начальная скорость КА
start_t = time.time()  # время начала отсчета интегрирования
res_p = RK4(f, p0, dt, h_lim, mu, sigma, R_km)  # вызов функции Рунге-Кутта
print("Интегрирование успешно")
print("Всего интегрирование заняло:", time.time() - start_t, "c")

# count = 20  # количество необходимых строк в таблице
# # Сортировка базы данных для вывода в файл
# x_sort, y_sort, z_sort, vx_sort, vy_sort, vz_sort, t_sort = [], [], [], [], [], [], []
# sort(res_p, count, x_sort, y_sort, z_sort, vx_sort, vy_sort, vz_sort, t_sort)
# # Вывод в файл при помощи модуля pandas
# file = pd.DataFrame({"t": t_sort,
#                      "x": x_sort,
#                      "y": y_sort,
#                      "z": z_sort,
#                      "Vx": vx_sort,
#                      "Vy": vy_sort,
#                      "Vz": vz_sort})
# file.to_excel("./results100kmGOST.xlsx")
height = h_to_km(R_km)

plt.figure(1, figsize=(9, 3))
plt.subplot(111)
plt.title("Высота")  # Построение графика h от t
plt.xlabel("t, сек")
plt.ylabel("h, км")
plt.plot(res_p[7], height)
resultplot = plt.figure().add_subplot(projection="3d", elev=45, azim=135)
plt.title("Траектория")  # Построение 3d траектории
plt.xlabel("x, км")
plt.ylabel("y, км")
resultplot.set_zlabel("z, км")
resultplot.plot(res_p[0], res_p[1], res_p[2])
plt.show()
