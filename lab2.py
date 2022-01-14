import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Сircle1(X, Y):
    CX = [X + R * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + R * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

def Сircle2(X, Y):
    CX = [X + r * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + r * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

def Сircle3(X, Y):
    CX = [X + r * 0.3 * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + r * 0.3 * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

def anima(i):
    Beam_1.set_data([X20[i], X30[i]], [Y20[i], Y30[i]])
    Beam_2.set_data([X10[i], X20[i]], [Y10[i], Y20[i]])
    Beam_3.set_data([X20[i], X20[i]], [Y20[i], -0.9])
    circle1.set_data(*Сircle1(X10[i], Y10[i]))
    circle2.set_data(*Сircle2(X20[i], Y20[i]))
    circle3.set_data(*Сircle3(X30[i], Y30[i]))
    return circle1, circle2, circle3, Beam_1, Beam_2, Beam_3


def diffur():
    dy = sp.zeros(4)
    return dy

t = sp.Symbol('t')
m1 = 2.0
m2 = 1.0
r = 0.1
R = 0.5
l = 0.5
g = 9.81
omega0 = math.pi/4
domega0 = 0
phi0 = 0
dphi0 = 0
# phi = sp.sin(math.pi/6*t)
phi = sp.sin(math.pi/6*t)
dphi = sp.diff(phi, t)
ddphi = sp.diff(dphi, t)
# omega = sp.sin(math.pi/4*t)
omega = sp.sin(math.pi/4*t)
domega = sp.diff(omega, t)
ddomega = sp.diff(domega, t)

x2 = (R-r) * sp.sin(ddomega)
y2 = -(R-r) * sp.cos(ddomega)
vx2 = sp.diff(x2, t)
vy2 = sp.diff(y2, t)
ax2 = sp.diff(vx2, t)
ay2 = sp.diff(vy2, t)

# x3 = x2 + l * sp.sin(phi)
# y3 = y2 - l * sp.cos(phi)
x3 = x2 + l * sp.sin(ddphi * sp.cos(phi-omega) + dphi ** 2 * sp.sin(phi-omega))
y3 = y2 - l * sp.cos(ddphi * sp.cos(phi-omega) + dphi ** 2 * sp.sin(phi-omega))
vx3 = sp.diff(x3, t)
vy3 = sp.diff(y3, t)
ax3 = sp.diff(vx3, t)
ay3 = sp.diff(vy3, t)

T = np.linspace(0, 50, 800)
X10 = np.zeros_like(T)
Y10 = np.zeros_like(T)
X20 = np.zeros_like(T)
Y20 = np.zeros_like(T)
X30 = np.zeros_like(T)
Y30 = np.zeros_like(T)

V = np.zeros_like(T)


x = np.linspace(0, 0, 800)
y = np.linspace(-0.6, 0, 800)

for i in np.arange(len(T)):
    X20[i] = sp.Subs(x2, t, T[i])
    Y20[i] = sp.Subs(y2, t, T[i])
    X30[i] = sp.Subs(x3, t, T[i])
    Y30[i] = sp.Subs(y3, t, T[i])

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
ax1.plot(x, y, linestyle = '--', linewidth = 1, color = 'grey')

ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, X20)
plt.title('Vx of the O1')
plt.xlabel('t values')
plt.ylabel('Vx values')

ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, Y20)
plt.title('Vy of the O1')
plt.xlabel('t values')
plt.ylabel('Vy values')

ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(T, X30)
plt.title('Vx of the Point A')
plt.xlabel('t values')
plt.ylabel('Vx values')

ax5 = fig.add_subplot(4, 2, 8)
ax5.plot(T, Y30)
plt.title('Vy of the Point A')
plt.xlabel('t values')
plt.ylabel('Vy values')

plt.subplots_adjust(wspace=0.3, hspace=0.7)

Beam_1, = ax1.plot([X20[0], X20[0] + l * sp.sin(math.pi)], [Y20[0], Y20[0] + l * sp.cos(math.pi)], 'black')
Beam_2, = ax1.plot([X10[0], X20[0]], [Y10[0], Y20[0]], linestyle = '--', linewidth = 1, color = 'grey')
Beam_3, = ax1.plot([X20[0], X20[0]], [Y20[0], y[0]], linestyle = '--', linewidth = 1, color = 'grey')
circle1, = ax1.plot(*Сircle1(X10[0], Y10[0]), 'black')
circle2, = ax1.plot(*Сircle2(X20[0], Y20[0]), 'black')
circle3, = ax1.plot(*Сircle3(X30[0], Y30[0]), 'black')

anim = FuncAnimation(fig, anima, frames=800, interval=10, blit=True)
plt.show()
