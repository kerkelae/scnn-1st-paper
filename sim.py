import math

from dipy.core.gradients import gradient_table
import numpy as np
import torch

from sh import l0s, ls, sft, vertices


rf_btens = torch.tensor(
    gradient_table(np.ones(len(vertices)), vertices, btens="LTE").btens
)
sim_sft = sft.double()


def compartment_model_simulation(bval, bvecs_isft, ads, rds, fs, odfs, device):
    """Simulate diffusion-weighted MR measurements."""
    n_simulations = ads.shape[0]
    n_compartments = ads.shape[1]
    Ds = torch.zeros(n_simulations, n_compartments, 3, 3).to(device)
    Ds[:, :, 2, 2] = ads  # aligned with the z-axis
    Ds[:, :, 1, 1] = rds
    Ds[:, :, 0, 0] = rds
    response = torch.sum(
        fs.to(device).unsqueeze(2)
        * torch.exp(
            -torch.sum(
                bval * rf_btens.unsqueeze(0).unsqueeze(1).to(device) * Ds.unsqueeze(2),
                dim=(3, 4),
            )
        ),
        dim=1,
    )
    response_sh = (sim_sft.to(device) @ response.unsqueeze(-1)).squeeze(-1)
    convolution_sh = (
        torch.sqrt(4 * math.pi / (2 * ls.to(device) + 1))
        * odfs.to(device)
        * response_sh[:, l0s]
    )
    simulated_signals = bvecs_isft.to(device) @ convolution_sh.unsqueeze(-1)
    return simulated_signals
