import osqp
import osqppurepy as osqppurepy
import scipy.sparse as sparse
import scipy as sp
import numpy as np

sp.random.seed(6)

n = 2000
m = 3000

# Add constraints
A = sparse.random(m, n, density=0.4, format='csc')
l = -sp.rand(m)
u = sp.rand(m)

random_scaling = np.power(10, 5*np.random.randn())
P = random_scaling * sparse.random(n, n, density=0.4)
P = P.dot(P.T).tocsc()
q = sp.randn(n)

osqp_opts = {'adaptive_rho_interval': 100,  # to make C code not depend on timing
             'check_termination': 1,  # Check termination every iteration
             }


# OSQP
model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res = model.solve()

# OSQPPUREPY
model = osqppurepy.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, linsys_solver=2, **osqp_opts)
res_purepy = model.solve()


print("Norm difference x OSQP and OSQPPUREPY %.4e" %
      np.linalg.norm(res.x - res_purepy.x))
print("Norm difference y OSQP and OSQPPUREPY %.4e" %
      np.linalg.norm(res.y - res_purepy.y))
