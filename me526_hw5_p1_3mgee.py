"""
me 526 hw5 p1 by michelle gee

1. Consider the mass diffusion equation:
∂2c/∂x2 + ∂2c/∂y2 = 0; 0 ≤ x ≤ 1, 0 ≤ y ≤ 1, (1)
with boundary conditions:
c(x, 0) = 0; c(x, 1) = 0; ∂c/∂x∣x = 1 = 0; (2)
and
c(0, y) = 1; 0.4 ≤ y ≤ 0.6 (3)
        = 0; 0 ≤ y < 0.4, 0.6 < y ≤ 1 (4)

(a) Using uniform grids of spacing h with N segments in both x and y directions, obtain the finite difference
approximation for the above equation using central differencing.
(b) Solve the system of equations using point-Jacobi, Gauss-Seidel, and SOR (with overrelaxation factor of 1.7 and 1.9).
Use N = 40 (i.e. 41 points).
(c) Plot the solution converged solution from each method. Assume the solution converges when c at every x and y
location does not change by more than 0.001 between successive iterations.
(d) Tabulate the number of iterations needed for each method.
(e) Repeat the process for N = 80

"""

# import packages ------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# define functions -----------------------------------------------------------------------------------------------------
# initialize grid
def initialize_grid(N):
    c = np.zeros((N, N))
    c[:, 0] = 1.0  # BC c(0, y) = 1
    c[:int(0.4 * N), :] = 0.0  # BC c(x, y) = 0 for y < 0.4
    c[int(0.6 * N):, :] = 0.0  # BC c(x, y) = 0 for y > 0.6
    return c

# point-jacobi function
def point_jacobi(c, N, h, max_iterations=10000, tolerance=0.001):
    c_new = np.copy(c)
    iterations = 0
    while iterations < max_iterations:
        max_diff = 0.0 # initialize for convergence check
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                c_new[i, j] = 0.25 * (c[i + 1, j] + c[i - 1, j] + c[i, j + 1] + c[i, j - 1])
                diff = abs(c_new[i, j] - c[i, j])
                if diff > max_diff:
                    max_diff = diff
        c[:] = c_new[:]
        if max_diff < tolerance: # check for convergence
            break
        iterations += 1
    return c, iterations

# gauss-seidel function
def gauss_seidel(c, N, h, max_iterations=10000, tolerance=0.001):
    iterations = 0
    while iterations < max_iterations:
        max_diff = 0.0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                c_old = c[i, j]
                c[i, j] = 0.25 * (c[i + 1, j] + c[i - 1, j] + c[i, j + 1] + c[i, j - 1])
                diff = abs(c[i, j] - c_old)
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tolerance:
            break
        iterations += 1
    return c, iterations

# successive over relaxation (SOR) function
def sor(c, N, h, omega, max_iterations=10000, tolerance=0.001):
    iterations = 0
    while iterations < max_iterations:
        max_diff = 0.0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                c_old = c[i, j]
                c[i, j] = (1 - omega) * c[i, j] + (omega / 4) * (
                    c[i + 1, j] + c[i - 1, j] + c[i, j + 1] + c[i, j - 1])
                diff = abs(c[i, j] - c_old)
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tolerance:
            break
        iterations += 1
    return c, iterations

# the one function to rule them all (call methods and plot)
def solve_and_plot(N, h):
    c = initialize_grid(N)

    # call point-jacobi method
    c_jacobi, iter_jacobi = point_jacobi(np.copy(c), N, h)

    # call gauss-seidel method
    c_gauss_seidel, iter_gauss_seidel = gauss_seidel(np.copy(c), N, h)

    # call SOR method
    omega_1 = 1.7
    c_sor_1, iter_sor_1 = sor(np.copy(c), N, h, omega_1)
    omega_2 = 1.9
    c_sor_2, iter_sor_2 = sor(np.copy(c), N, h, omega_2)

    # plot it all on a colormap since we have 2d
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.title("1e, N = 80: Point-Jacobi")
    plt.imshow(c_jacobi, cmap='viridis')
    plt.colorbar()

    plt.subplot(222)
    plt.title("1e, N = 80: Gauss-Seidel")
    plt.imshow(c_gauss_seidel, cmap='viridis')
    plt.colorbar()

    plt.subplot(223)
    plt.title("1e, N = 80: SOR, lambda = 1.7")
    plt.imshow(c_sor_1, cmap='viridis')
    plt.colorbar()

    plt.subplot(224)
    plt.title("1e, N = 80: SOR, lambda = 1.9")
    plt.imshow(c_sor_2, cmap='viridis')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # print # iterations for each method
    print(f"iterations for point-jacobi, N = 80: {iter_jacobi}")
    print(f"iterations for Gauss-Seidel, N = 80: {iter_gauss_seidel}")
    print(f"iterations for SOR (1.7), N = 80: {iter_sor_1}")
    print(f"iterations for SOR (1.9), N = 80: {iter_sor_2}")

# grid initialization and plotting--------------------------------------------------------------------------------------
N = 81
h = 1.0 / (N - 1)

# compute and plot
solve_and_plot(N, h)
print("complete")
