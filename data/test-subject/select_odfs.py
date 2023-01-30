import healpy as hp
import nibabel as nib
import numpy as np
from scipy.special import sph_harm


l_max = 8
n_coeffs = int(0.5 * (l_max + 1) * (l_max + 2))


def sh(l, m, thetas, phis):
    """Real and symmetric spherical harmonic basis function."""
    if l % 2 == 1:
        return np.zeros(len(phis))
    if m < 0:
        return np.sqrt(2) * sph_harm(-m, l, phis, thetas).imag
    if m == 0:
        return sph_harm(m, l, phis, thetas).real
    if m > 0:
        return np.sqrt(2) * sph_harm(m, l, phis, thetas).real


n_sides = 2**5  # high sampling density
vertices = np.vstack(hp.pix2vec(n_sides, np.arange(12 * n_sides**2))).T
thetas = np.arccos(vertices[:, 2])
phis = np.arctan2(vertices[:, 1], vertices[:, 0]) + np.pi
isft = np.zeros((len(vertices), n_coeffs))
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        isft[:, int(0.5 * l * (l + 1) + m)] = sh(l, m, thetas, phis)
sft = np.linalg.pinv(isft.T @ isft) @ isft.T

mask = nib.load("brain_mask.nii.gz").get_fdata().astype(bool)
odfs_sh = nib.load("wmfod.nii.gz").get_fdata()[mask]
odfs_sh = odfs_sh / odfs_sh[:, 0][:, np.newaxis] / np.sqrt(4 * np.pi)  # normalize

for i in range(0, len(odfs_sh), int(1e4)):

    idx = np.arange(i, i + int(1e4))
    if idx.max() >= len(odfs_sh):
        idx = np.arange(i, len(odfs_sh))

    ss_odfs_sh = odfs_sh[idx]

    ss_odfs = (isft @ ss_odfs_sh[..., np.newaxis])[..., 0]
    idx = ~(np.sum(np.isnan(ss_odfs), axis=1).astype(bool)) & (
        np.min(ss_odfs, axis=1) > -1e-2
    )  # exclude ODFs with significantly negative values
    ss_odfs_sh = ss_odfs_sh[idx]

    if i == 0:
        filtered_odfs_sh = ss_odfs_sh
    else:
        filtered_odfs_sh = np.vstack((filtered_odfs_sh, ss_odfs_sh))

print(len(odfs_sh))
print(len(filtered_odfs_sh))
np.savetxt("odfs_sh.txt", filtered_odfs_sh)
