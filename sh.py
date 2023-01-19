import math

import healpy as hp
import numpy as np
from scipy.special import sph_harm
import torch


l_max = 10
n_coeffs = int(0.5 * (l_max + 1) * (l_max + 2))

ls = torch.zeros(n_coeffs, dtype=int)
l0s = torch.zeros(n_coeffs, dtype=int)
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        ls[int(0.5 * l * (l + 1) + m)] = l
        l0s[int(0.5 * l * (l + 1) + m)] = int(0.5 * l * (l + 1))


def sh(l, m, thetas, phis):
    """Real and symmetric spherical harmonic basis function."""
    if l % 2 == 1:
        return torch.zeros(len(phis))
    if m < 0:
        return math.sqrt(2) * sph_harm(-m, l, phis, thetas).imag
    if m == 0:
        return sph_harm(m, l, phis, thetas).real
    if m > 0:
        return math.sqrt(2) * sph_harm(m, l, phis, thetas).real


n_sides = 2**3
vertices = torch.vstack(
    [torch.tensor(i) for i in hp.pix2vec(n_sides, np.arange(12 * n_sides**2))]
).T
thetas = torch.arccos(vertices[:, 2])
phis = torch.arctan2(vertices[:, 1], vertices[:, 0]) + math.pi
isft = torch.zeros((len(vertices), n_coeffs))
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        isft[:, int(0.5 * l * (l + 1) + m)] = sh(l, m, thetas, phis)
sft = torch.linalg.pinv(isft.T @ isft) @ isft.T
