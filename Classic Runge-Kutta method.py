import matplotlib.pyplot as plt

h, T = 10**-3, 100
s, l, b = 10, 28, 8 / 3
x_list, y_list, z_list = [1], [1], [1]
errors, t = [0], [0]


def Runge_Kutta(step):
    kx1 = s * (y_list[-1] - x_list[-1])
    ky1 = x_list[-1] * (l - z_list[-1]) - y_list[-1]
    kz1 = (x_list[-1] * y_list[-1] - b * z_list[-1])

    kx2 = s * (y_list[-1] + (ky1 * step) / 2 - x_list[-1] - (kx1 * step) / 2)
    ky2 = (x_list[-1] + (kx1 * step) / 2) * (l - z_list[-1] - (kz1 * step) / 2) - y_list[-1] - (ky1 * step) / 2
    kz2 = ((x_list[-1] + (kx1 * step) / 2) * (y_list[-1] + (ky1 * step) / 2) - b * (z_list[-1]) + (kz1 * step) / 2)

    kx3 = s * (y_list[-1] + (ky2 * step) / 2 - x_list[-1] - (kx2 * step) / 2)
    ky3 = (x_list[-1] + (kx2 * step) / 2) * (l - z_list[-1] - (kz2 * step) / 2) - y_list[-1] - (ky2 * step) / 2
    kz3 = ((x_list[-1] + (kx2 * step) / 2) * (y_list[-1] + (ky2 * step) / 2) - b * (z_list[-1]) + (kz2 * step) / 2)

    kx4 = s * (y_list[-1] + (ky3 * step) - x_list[-1] - (kx3 * step))
    ky4 = (x_list[-1] + (kx3 * step)) * (l - z_list[-1] - (kz3 * step)) - y_list[-1] - (ky3 * step)
    kz4 = ((x_list[-1] + (kx3 * step)) * (y_list[-1] + (ky3 * step)) - b * (z_list[-1]) + (kz3 * step))

    return [x_list[-1] + (step * (kx1 + 2 * kx2 + 2 * kx3 + kx4)) / 6,
            y_list[-1] + (step * (ky1 + 2 * ky2 + 2 * ky3 + ky4)) / 6,
            z_list[-1] + (step * (kz1 + 2 * kz2 + 2 * kz3 + kz4)) / 6]


print("Initial conditions: t = " + str(t[0]))
print("                    (x; y; z) = (" + str(x_list[0]) + "; " + str(y_list[0]) + "; " + str(z_list[0]) + ")")
print("σ = " + str(s))
print("λ = " + str(l))
print("β = " + str(b) + "\n")

print("Calculation started...")
while t[-1] <= T:
    solve_whole = Runge_Kutta(2 * h)

    solve_half = Runge_Kutta(h)
    x_list.append(solve_half[0])
    y_list.append(solve_half[1])
    z_list.append(solve_half[2])

    solve_half = Runge_Kutta(h)
    x_list[-1] = solve_half[0]
    y_list[-1] = solve_half[1]
    z_list[-1] = solve_half[2]

    errors.append(max(abs(solve_half[0] - solve_whole[0]) / 15,
                      abs(solve_half[1] - solve_whole[1]) / 15,
                      abs(solve_half[2] - solve_whole[2]) / 15))
    t.append(t[-1] + h)

    if (t[-1] >= T / 4) & (t[-2] < T / 4):
        print("25% . . .")
    elif (t[-1] >= T / 2) & (t[-2] < T / 2):
        print("50% . . .")
    elif (t[-1] >= 3 * T / 4) & (t[-2] < 3 * T / 4):
        print("75% . . .")
print("System integration finished!\n")

ax = plt.axes(projection="3d")
ax.plot(x_list, y_list, z_list, linewidth=0.5, color='r')
plt.title("Phase portrait")
plt.show()

plt.title("Local error estimation graph")
plt.plot(t, errors)
plt.show()
