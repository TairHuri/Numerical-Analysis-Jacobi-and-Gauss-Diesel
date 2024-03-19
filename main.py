import numpy as np
from numpy.linalg import norm

def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(n) if j != i):
            return False
    return True

def jacobi_iterative(A, b, X0, TOL=1e-3, N=200):
    n = len(A)
    k = 1
    while k <= N:
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * X0[j]
            x[i] = (b[i] - sigma) / A[i][i]

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)
        k += 1
        X0 = x.copy()
    raise ValueError("Maximum number of iterations exceeded")

def gauss_seidel(A, b, X0, TOL=1e-3, N=200):
    n = len(A)
    k = 1
    x = np.zeros(n, dtype=np.double)
    while k <= N:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)
        k += 1
        X0 = x.copy()
    raise ValueError("Maximum number of iterations exceeded")

def solve_linear_system(A, b, method='jacobi', X0=None, TOL=1e-3, N=200):
    n = len(A)
    if not (len(A) == len(A[0]) == 3):
        raise ValueError("The matrix is not 3x3")
    if method == 'jacobi':
        if not is_diagonally_dominant(A):
            raise ValueError("A matrix has no dominant diagonal.")
        print('Matrix is diagonally dominant - performing Jacobi algorithm\n')
        print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, n + 1)]]))
        print("-----------------------------------------------------------------------------------------------")
        if X0 is None:
            X0 = np.zeros(n, dtype=np.double)
        return jacobi_iterative(A, b, X0, TOL, N)
    elif method == 'gauss_seidel':
        if not is_diagonally_dominant(A):
            raise ValueError("A matrix has no dominant diagonal.")
        print('Matrix is diagonally dominant - performing Gauss-Seidel algorithm\n')
        print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, n + 1)]]))
        print("-----------------------------------------------------------------------------------------------")
        if X0 is None:
            X0 = np.zeros(n, dtype=np.double)
        return gauss_seidel(A, b, X0, TOL, N)
    else:
        raise ValueError("Unsupported method. Choose either 'jacobi' or 'gauss_seidel'.")

if __name__ == "__main__":
    A = np.array([[5, 3, 0], [3, 11, 5], [0, 5, 6]], dtype=np.double)
    b = np.array([8, 19, 11], dtype=np.double)
    try:
        X = solve_linear_system(A, b, method='jacobi')
        print("Jacobi Method:")
        print("Solution:", X)

        print("\nGauss-Seidel Method:")
        X = solve_linear_system(A, b, method='gauss_seidel')
        print("Solution:", X)

    except ValueError as e:
        print(str(e))
