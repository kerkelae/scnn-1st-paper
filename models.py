import math

import torch

from sh import l_max, n_coeffs, sft, isft


class MLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(120, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        return x


class PAMLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = torch.vstack(
            (torch.mean(x[:, 0:60], dim=1), torch.mean(x[:, 60::], dim=1))
        ).T
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        return x


class SHMLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2 * n_coeffs, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = x.reshape(-1, 2 * n_coeffs)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        return x


class SphConv(torch.nn.Module):
    def __init__(self, l_max, c_in, c_out):
        super().__init__()
        self.l_max = l_max
        self.c_in = c_in
        self.c_out = c_out
        ls = torch.zeros(n_coeffs, dtype=int)
        for l in range(0, l_max + 1, 2):
            for m in range(-l, l + 1):
                ls[int(0.5 * l * (l + 1) + m)] = l
        self.register_buffer("ls", ls)
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.c_out, self.c_in, int(self.l_max / 2) + 1)
        )
        torch.nn.init.uniform_(self.weights)

    def forward(self, x):
        weights_exp = self.weights[:, :, (self.ls / 2).long()]
        ys = torch.sum(
            torch.sqrt(
                math.pi / (2 * self.ls.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 1)
            )
            * weights_exp.unsqueeze(0)
            * x.unsqueeze(1),
            dim=2,
        )
        return ys


class SCNNModel(torch.nn.Module):
    def __init__(self, l_max):
        super().__init__()
        self.conv1 = SphConv(l_max, 2, 16)
        self.conv2 = SphConv(l_max, 16, 32)
        self.conv3 = SphConv(l_max, 32, 64)
        self.fc1 = torch.nn.Linear(64, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 2)
        self.register_buffer("sft", sft)
        self.register_buffer("isft", isft)

    def nonlinearity(self, x):
        return (
            self.sft @ torch.nn.functional.relu(self.isft @ x.unsqueeze(-1))
        ).squeeze(-1)

    def global_pooling(self, x):
        return torch.mean(self.isft @ x.unsqueeze(-1), dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.nonlinearity(x)
        x = self.conv3(x)
        x = self.nonlinearity(x)
        x = self.global_pooling(x)
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x
