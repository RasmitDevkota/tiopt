import numpy as np
# np.random.seed(1234)

from scipy.special import factorial, lpmv, sph_harm_y
from scipy.integrate import lebedev_rule

from itertools import permutations

################################################################################

def eval_cartesian_coords(f, x, y, z):
    if x >= 0 and x < NX and y >= 0 and y < NY and z >= 0 and z < NZ:
        if int(x) == x and int(y) == y and int(z) == z:
            f_val = f[int(x), int(y), int(z)]
        else:
            next_x, next_y, next_z = int(np.ceil(x)), int(np.ceil(y)), int(np.ceil(z))
            prev_x, prev_y, prev_z = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))

            w_x_n, w_y_n, w_z_n = next_x - x, next_y - y, next_z - z
            w_x_p, w_y_p, w_z_p = x - prev_x, y - prev_y, z - prev_z

            w_n = (w_x_n + w_y_n + w_z_n)/3
            w_p = (w_x_p + w_y_p + w_z_p)/3

            f_val = (1-w_n) * f[next_x, next_y, next_z] + (1-w_p) * f[prev_x, prev_y, prev_z]
    else:
        f_val = 0

    return f_val

def eval_sph_coords(f, r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    f_val = eval_cartesian_coords(f, x, y, z)

    return f_val

def cartesian_to_theta(x,y,z):
    if z > 0:
        return np.arctan(np.sqrt(x**2 + y**2)/z)
    elif z < 0:
        return np.arctan(np.sqrt(x**2 + y**2)/z) + np.pi
    elif z == 0:
        return np.pi/2

def cartesian_to_phi(x,y,z):
    if x > 0:
        return np.arctan(y/x)
    elif x < 0 and y >= 0:
        return np.arctan(y/x) + np.pi
    elif x < 0 and y < 0:
        return np.arctan(y/x) - np.pi
    elif x == 0 and y > 0:
        return np.pi/2
    elif x == 0 and y < 0:
        return -np.pi/2
    else:
        return 0

################################################################################

# def Y(l, m, costheta, phi):
#     return (-1)**m * np.sqrt((2*l+1)/(4*np.pi) * factorial(l-m)/factorial(l+m)) * lpmv(m, l,
#     costheta) * np.exp(1j*m*phi)

def Y(l, m, theta, phi):
    return sph_harm_y(l, m, theta, phi)

################################################################################

def lebedev_quadrature(f, n):
    x, w = lebedev_rule(n)

    points = [[float(u[0]), float(u[1]), float(u[2])] for u in zip(x[0], x[1], x[2])]

    num_points = np.size(x[0])

    num_weights = np.size(w)

    I = 0
    for pt in range(num_points):
        pt_x = x[0][pt]
        pt_y = x[1][pt]
        pt_z = x[2][pt]

        I += w[pt] * f(pt_x, pt_y, pt_z)

    I *= 4*np.pi

    return I

################################################################################

def compute_spherical_harmonics_expansion(data, l_max, n):
    data_lambda = lambda x,y,z : eval_cartesian_coords(data, x, y, z)

    alpha_lms = []

    for l in range(l_max+1):
        for m in range(-l, l+1):
            Ylm_lambda = lambda x,y,z : Y(l, m, cartesian_to_theta(x,y,z), cartesian_to_phi(x,y,z))

            alpha_lm = lebedev_quadrature(lambda x, y, z : data_lambda(x,y,z) * np.conj(Ylm_lambda(x,y,z)), n)

            assert not np.isnan(alpha_lm), f"alpha_{l},{m} is nan"

            alpha_lms.append(alpha_lm)

    assert len(alpha_lms) == sum(2*l+1 for l in range(l_max+1)), f"{len(alpha_lms)}, {list(2*l+1 for l in range(l_max+1))}"

    return alpha_lms

def evaluate_spherical_harmonics_representation(alpha_lms, l_max, data_shape):
    NX, NY, NZ = data_shape
    data = np.zeros((NX, NY, NZ))

    assert len(alpha_lms) == sum(2*l+1 for l in range(l_max+1)), f"{len(alpha_lms)}, {list(2*l+1 for l in range(l_max+1))}"

    for x in range(NX):
        for y in range(NY):
            for z in range(NZ):
                # R = 1/((x-NX/2)**2+(y-NY/2)**2+(z-NZ/2)**2)
                R = ((x)**2+(y)**2+(z)**2+1)

                c = 0
                for l in range(l_max+1):
                    for m in range(-l, l+1):
                        data[x,y,z] += R**l * alpha_lms[c] * Y(l, m, cartesian_to_theta(x,y,z), cartesian_to_phi(x,y,z))
                        c += 1

    return data

################################################################################

l_max = 10
n = 7

NX = 25
NY = 25
NZ = 25
# V = np.random.rand(NX, NY, NZ)
# V = np.arange(NX*NY*NZ).reshape((NX, NY, NZ))
# V = np.zeros((NX, NY, NZ))
# V = np.ones((NX, NY, NZ))

V = np.zeros((NX,NY,NZ))
for x in range(NX):
    for y in range(NY):
        for z in range(NZ):
            # V[x,y,z] = 1/((x-NX/2)**2+(y-NY/2)**2+(z-NZ/2)**2)
            V[x,y,z] = 1/((x)**2+(y)**2+(z)**2+1)

            # if x**2 + y**2 + z**2 == 25:
                # V[x,y,z] = 1

# x, w = lebedev_rule(n)
# print(w)
# points = [[float(u[0]), float(u[1]), float(u[2])] for u in zip(x[0], x[1], x[2])]
# V_sph = np.array([1/(u[0]**2+1) for u in points])

V_lms = compute_spherical_harmonics_expansion(V, l_max, n)
print(V_lms)

V_check = evaluate_spherical_harmonics_representation(V_lms, l_max, (NX, NY, NZ))
print(V_check)

print(np.linalg.norm(V_check - V)/np.size(V))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2)

ax[0].imshow(V[:,:,0])
ax[1].imshow(V_check[:,:,0])

plt.show()

