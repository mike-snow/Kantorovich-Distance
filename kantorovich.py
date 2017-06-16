import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


# solution to the Kantorovich optimal transport problem on domain [0,1].


m = 16
n = 16

source = matrix([0]*(m-1) + [10])
target = matrix([8] + [0]*(n-1))

# Normalise masses s.t. \sum = 1


def normalise(mass):
    norm_mass = mass/float(sum(mass))
    return norm_mass


# Coefficient matrix for LP problem


def coeff_matrix(m, n):
    a = np.c_[np.kron(np.eye(m), [1]*m), np.ones(n)]
    b = np.c_[np.kron([1]*n, np.eye(m)), np.zeros(m)]
    A = matrix(np.r_[a, b])
    return A

# create Euclidean cost matrix


v1 = []
v2 = []


for i in range(0, m):
    v1.append(i/float(m-1))

for j in range(0, n):
    v2.append(j/float(n-1))

cost = []

for ii in v1:
    for jj in v2:
        cost.append((ii-jj)**2)

cost.append(0.0)

# Set up LP problem & solve 

A = coeff_matrix(m, n)

b = matrix(np.r_[normalise(source), normalise(target)])

c = matrix(cost)

G = matrix(-np.eye(n*m+1)) 

h = matrix(-np.zeros(n*m+1)) # Ensure x \geq 0

solution = solvers.lp(c, G, h, A, b, solver='solver')

x = solution['x']

# Approximation to Wasserstein distance

print x.trans()*c

# plots of distributions

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(v1, normalise(source), s=20, c='b', marker="s", label='first')
ax1.scatter(v2, normalise(target), s=20, c='r', marker="o", label='second')
plt.legend(loc='upper left')


plt.show()
