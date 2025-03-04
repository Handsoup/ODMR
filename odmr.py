import numpy as np
import math
import matplotlib.pyplot as plt

MHztoKelvin=float(0.000047991939324)
GHztoKelvin=float(0.047991939324)
gausstokelvin=float(0.0000671714)

def rot(n1, n2):
    n1 = np.array(n1)
    n2 = np.array(n2)

    # Normalize the input vectors
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    u = np.cross(n1, n2)
    c = np.dot(n1, n2)
    s = np.linalg.norm(u)

    if c == 1 or s == 0:
        return np.identity(3)
    else:
        u = u / s
        asym = np.array([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ])
        diag = np.eye(3)
        rotation_matrix = c * diag + s * asym + (1 - c) * np.outer(u, u)
        return rotation_matrix


def cartesian_to_spherical(x, y, z):
    # Compute the radial distance
    r = math.sqrt(x**2 + y**2 + z**2)

    # Compute the polar angle (theta)
    theta = math.acos(z / r)

    # Compute the azimuthal angle (phi)
    phi = math.atan2(y, x)

    return theta, phi

# Spin operators
s_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
s_y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2.0)
s_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2.0)

# Square of the spin operators
s_y_2 = np.array([[1, 0, -1], [0, 2, 0], [-1, 0, 1]]) / 2.0
s_x_2 = np.dot(s_x, s_x)
s_z_2 = np.dot(s_z, s_z)

# Spin vector with Hermitian cartesian component operators
s = np.array([s_x, s_y, s_z])

# Initialize empty lists for results
s_1, s_2, s_3 = [], [], []
eigvectors0, eigvectors1, eigvectors2 = [], [], []

# Magnetic field range
B = np.linspace(0, 150000, 150001)

# Constants
D = 2875
E = 0
g = float(2.0026)

# Directions
n_z = np.array([0, 0, -1])
#n_z = np.array([0, 1, 0])
#theta = 0.95531661812450927816385710251576
#phi = 0.78539816
name = "Figure_5_100_at"
theta, phi = cartesian_to_spherical(1,0,0)
direction = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))

dir1 = np.array([1, -1, -1]) / np.sqrt(3)
dir2 = np.array([-1, 1, -1]) / np.sqrt(3)
dir3 = np.array([-1, -1, 1]) / np.sqrt(3)
dir4 = np.array([1, 1, 1]) / np.sqrt(3)

H_nv = D * (s_z_2 - 1/3 * (s_x_2 + s_y_2 + s_z_2)) * MHztoKelvin + E * (s_x_2 - s_y_2) * MHztoKelvin

R_mat = rot(n_z, direction)

# Main computation
for diri in [dir1, dir2, dir3, dir4]:
    alpha = np.einsum('ab,b', R_mat, diri)
    H_z = g * np.einsum('a,abc->bc', alpha, s) * gausstokelvin
    print('aa')
    for b in B:
      H_zeeman = H_z * b
      H = H_zeeman + H_nv
      eigenValues, eigenVectors = np.linalg.eigh(H)
      idx = eigenValues.argsort()
      eigval = eigenValues[idx]
      eigvect = eigenVectors[:, idx]

      s_1.append(eigval[0])
      s_2.append(eigval[1])
      s_3.append(eigval[2])
      eigvectors0.append(eigvect[0])
      eigvectors1.append(eigvect[1])
      eigvectors2.append(eigvect[2])
      # eigvectors0 = np.append(eigvectors0, np.linalg.norm(eigvect[:, 0]))
      # eigvectors1 = np.append(eigvectors1, np.linalg.norm(eigvect[:, 1]))
      # eigvectors2 = np.append(eigvectors2, np.linalg.norm(eigvect[:, 2]))

      
s1=np.array(s_1).reshape(4,len(B))
s2=np.array(s_2).reshape(4,len(B))
s3=np.array(s_3).reshape(4,len(B))

plt.figure(figsize=(12, 8))
plt.plot(B, s1[0], label=f's1')
plt.plot(B, s2[0], label=f's2')
plt.plot(B, s3[0], label=f's3')
plt.xlim(0, 5000)
plt.ylim(-1,1)
plt.savefig(name + "0_short.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[0], label=f's1')
plt.plot(B, s2[0], label=f's2')
plt.plot(B, s3[0], label=f's3')
plt.savefig(name + "0_full.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[1], label=f's1')
plt.plot(B, s2[1], label=f's2')
plt.plot(B, s3[1], label=f's3')
plt.xlim(0, 5000)
plt.ylim(-1,1)
plt.savefig(name + "1_short.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[1], label=f's1')
plt.plot(B, s2[1], label=f's2')
plt.plot(B, s3[1], label=f's3')
plt.savefig(name + "1_full.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[2], label=f's1')
plt.plot(B, s2[2], label=f's2')
plt.plot(B, s3[2], label=f's3')
plt.xlim(0, 5000)
plt.ylim(-1,1)
plt.savefig(name + "2_short.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[2], label=f's1')
plt.plot(B, s2[2], label=f's2')
plt.plot(B, s3[2], label=f's3')
plt.savefig(name + "2_full.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[3], label=f's1')
plt.plot(B, s2[3], label=f's2')
plt.plot(B, s3[3], label=f's3')
plt.xlim(0, 5000)
plt.ylim(-1,1)
plt.savefig(name + "3_short.png")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(B, s1[3], label=f's1')
plt.plot(B, s2[3], label=f's2')
plt.plot(B, s3[3], label=f's3')
plt.savefig(name + "3_full.png")
plt.show()
plt.close()


