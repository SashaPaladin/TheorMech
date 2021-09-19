import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation
import sympy as sp

# T = np.linspace(1, 10, 1000)
t = sp.Symbol('t')
R = 4
# Omega = 1
# x = R * (Omega * t - sp.sin(Omega * t))
# y = R * (1 - sp.cos(Omega * t))
r = 1 + sp.cos(t)
phi = 1.25 * t
x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
v = sp.sqrt(Vx * Vx + Vy * Vy)
a = sp.sqrt(Ax * Ax + Ay * Ay)
a_tan = sp.diff(v, t)
a_norm = sp.sqrt(a * a - a_tan * a_tan)
rho = v * v / a_norm
# ax = sp.diff(sp.diff(r, t), t) * sp.cos(phi) - 2 * sp.diff(r, t) * sp.sin(phi) - r * sp.cos(phi)
# ay = sp.diff(sp.diff(r, t), t) * sp.sin(phi) + 2 * sp.diff(r, t) * sp.cos(phi) - r * sp.sin(phi)
T = np.linspace(0, 50, 2000)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)  # кол-во элементов такое же, как в T
AX = np.zeros_like(T)
AY = np.zeros_like(T)
ANX = np.zeros_like(T)
ANY = np.zeros_like(T)
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])  # t == T[i] (грубо)
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    # ANX[i] = sp.Subs(ax, t, T[i])
    # ANY[i] = sp.Subs(ay, t, T[i])
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)  # будет 1 график
ax1.axis('equal')
# ax1.set(xlim=[-R, 12 * R], ylim=[-R, 4 * R])  # границы рисунка
ax1.set(xlim=[-R, R], ylim=[-R, R])  # границы рисунка
ax1.plot(X, Y)  # заполняем рисунок значениями массивов Х и У
P, = ax1.plot(X[0], Y[0], marker='o')  # рисуем точку
Vline, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')  # рисуем стрелочку красного цвета
Aline, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'g')  # рисуем стрелочку зеленого цвета


# ANline, = ax1.plot([X[0], X[0] + ANX[0]], [Y[0], Y[0] + ANY[0]], 'y')


def Rot2D(X, Y, Alpha):  # у стрелочки появился наконечник, правильно движется
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)  # координаты стрелочек
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)  # координаты стрелочек
    return RX, RY


ArrowX = np.array([-0.05 * R, 0, -0.05 * R])  # что-то происходит с координатами стрелочки
ArrowY = np.array([0.05 * R, 0, -0.05 * R])
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))  # рисуем стрелочку
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0], 'r')  # рисуем стрелочку
ARArrowX, ARArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))  # рисуем стрелочку ускорения
AArrow, = ax1.plot(ARArrowX + X[0] + AX[0], ARArrowY + Y[0] + AY[0], 'g')  # рисуем стрелочку ускорения


def anima(j):  # анимация движения стрелочки
    P.set_data(X[j], Y[j])
    Vline.set_data([X[j], X[j] + VX[j]], [Y[j], Y[j] + VY[j]])
    Aline.set_data([X[j], X[j] + AX[j]], [Y[j], Y[j] + AY[j]])
    # ANline.set_data([X[j], X[j] + ANX[j]], [Y[j], Y[j] + ANY[j]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[j], VX[j]))
    VArrow.set_data(RArrowX + X[j] + VX[j], RArrowY + Y[j] + VY[j])
    ARArrowX, ARArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[j], AX[j]))
    AArrow.set_data(ARArrowX + X[j] + AX[j], ARArrowY + Y[j] + AY[j])
    return P, Vline, VArrow, Aline, AArrow  # , ANline


anim = FuncAnimation(fig, anima, frames=1000, interval=2, blit=True)
plt.show()
