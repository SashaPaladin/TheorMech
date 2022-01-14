import math
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def odesys(y, t, m1, m2, l, g, r, R):

    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = m2 * l * np.cos(y[0] - y[1])
    a12 = ((3 / 2) * m1 + m2) * (R - r)
    a21 = l
    a22 = (R - r) * np.cos(y[0] - y[1])

    b1 = -(m1 + m2) * g * np.sin(y[1]) + m2 * l * y[2] ** 2 * np.sin(y[0] - y[1])
    b2 = -g * np.sin(y[0]) - (R-r) * y[3] ** 2 * np.sin(y[0] - y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return dy


l = 0.5
m1 = 2
m2 = 1
r = 0.1
R = 0.5
g = 9.81

t_fin = 20

T = np.linspace(0, t_fin, 1000)

phi0 = 0
thetta0 = math.pi / 4
dphi0 = 0
dthetta0 = 0

y0 = [phi0, thetta0, dphi0, dthetta0]

Y = odeint(odesys, y0, T, (m1, m2, l, g, r, R))

print(Y.shape)

phi = Y[:, 0]
thetta = Y[:, 1]


def Circle1(X, Y, radius):
    CX = [X + radius * math.cos(i / 100) for i in range(314, 628)]
    CY = [Y + radius * math.sin(i / 100) for i in range(314, 628)]
    return CX, CY
def Circle2(X, Y, radius):
    CX = [X + radius * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + radius * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

t = np.linspace(0, 10, 1001)

X_O = 0
Y_O = 0
X_1 = (R - r) * np.sin(thetta)
Y_1 = -(R - r) * np.cos(thetta)
X_A = X_1 + l * np.sin(phi)
Y_A = Y_1 - l * np.cos(phi)

VXA = np.diff(X_A)
VYA = np.diff(Y_A)
WXA = np.diff(VXA)
WYA = np.diff(VYA)

fig = plt.figure(figsize=[7, 5])
ax = fig.add_subplot(1, 2, 1)
ax.axis('equal')
ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

Point_O = ax.plot(X_O, Y_O, marker='o', color='black')[0]
Point_A = ax.plot(X_A, Y_A, marker='o', color='black')[0]
Point_1 = ax.plot(X_1, Y_1, marker='o', color='black', markersize=3)[0]
Line_O1A = ax.plot([X_1[0], X_A[0]], [Y_1[0], Y_A[0]], color='black')[0]
circle1, = ax.plot(*Circle2(X_1[0], Y_1[0], r), 'red')
circle2, = ax.plot(*Circle1(X_O, Y_O, R), 'red')


ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(VXA)
plt.title('Vx of ball')
plt.xlabel('t values')
plt.ylabel('Vx values')

ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(VYA)
plt.title('Vy of ball')
plt.xlabel('t values')
plt.ylabel('Vy values')

ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(WXA)
plt.title('Wx of ball')
plt.xlabel('t values')
plt.ylabel('Wy values')

ax5 = fig.add_subplot(4, 2, 8)
ax5.plot(WYA)
plt.title('Wy of ball')
plt.xlabel('t values')
plt.ylabel('Wx values')

plt.subplots_adjust(wspace=0.3, hspace=0.7)


def Anime(i):
    circle1.set_data(*Circle2(X_1[i], Y_1[i], r))
    circle2.set_data(*Circle1(X_O, Y_O, R))
    Point_O.set_data(X_O, Y_O)
    Point_1.set_data(X_1[i], Y_1[i])
    Point_A.set_data(X_A[i], Y_A[i])
    Line_O1A.set_data([X_1[i], X_A[i]], [Y_1[i], Y_A[i]])
    return [circle1, Point_O, Point_A, Line_O1A, circle2, Point_1]


anim = FuncAnimation(fig, Anime, frames=1000, interval=10)

plt.show()