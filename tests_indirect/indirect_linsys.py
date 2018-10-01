# Solve linear system indirectly using CG
import numpy as np
import scipy as sp
import scipy.sparse as spa
from scipy.sparse.linalg import spsolve

n = 100

# Create random positive semidefinite A
A = spa.random(n, n, density=0.4, format='csc')
A = A.dot(A.T).tocsc() + 0.1 * spa.eye(n)

# Create random rhs
b = sp.rand(n)

# Solve linear system with direct method
x_direct = spsolve(A, b)


def solve_cg(A, b, tol=1e-03, x=0, max_iters=100):
    if x == 0:
        x = np.zeros(b.size)

    # Compute inverse preconditoner M
    invM = spa.diags(np.reciprocal(A.diagonal()))

    # Initialize algorithm
    r = A.dot(x) - b
    y = invM.dot(r)  # M * y = r
    p = -y
    for k in range(max_iters):
        # Perform CG iterations
        ry = r.dot(y)
        Ap = A.dot(p)
        alpha = ry / (p.dot(Ap))
        x = x + alpha * p
        r = r + alpha * Ap
        y = invM.dot(r)
        ry_new = r.dot(y)
        beta = ry_new / ry
        p = -y + beta * p

        # Check if residual is less than tolerance
        norm_r = np.linalg.norm(r)
        if norm_r < tol:
            return x, k

    print("CG did not converge within %i iterations, residual %.2e > tolerance %.2e\n" % (max_iters, norm_r, tol))


# Solve with CG
x_cg, cg_iter = solve_cg(A, b)

print("|| x_cg - x_direct ||_2 = %.2e\n"
      % np.linalg.norm(x_cg - x_direct))
