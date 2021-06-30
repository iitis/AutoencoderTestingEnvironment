"""
Copyright 2021 Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
Authors:
- Kamil Książek (ITAI PAS, ORCID ID: 0000−0002−0201−6220),
- Przemysław Głomb (ITAI PAS, ORCID ID: 0000−0002−0215−4674),
- Michał Romaszewski (ITAI PAS, ORCID ID: 0000−0002−8227−929X),
- Michał Cholewa (ITAI PAS, ORCID ID: 0000−0001−6549−1590),
- Bartosz Grabowski (ITAI PAS, ORCID ID: 0000−0002−2364−6547)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

---

Autoencoders testing environment (ATE) v.1.0

Related to the work:
Stable training of autoencoders for hyperspectral unmixing

Source code for the review process of the 28th International Conference
on Neural Information Processing (ICONIP 2021)

"""

import torch

from torch import nn
from ray import tune

from util_nn import (sum_to_one_constraint, DynamicalSoftThresholding)

# ------------------------CONST ------------------------------------------------

#class name of the autoencoder class (you can leave it as it is)
AA_CLASS_NAME = 'Autoencoder'

# ------------------------AA CLASS------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, n_bands, n_endmembers):
        """
        Simple autoencoder (with relu)

        Parameters
        n_bands: no. bands
        n_endmembers: no. endmembers
        """
        super(Autoencoder, self).__init__()
        self.bands = n_bands
        self.endmembers = n_endmembers
        self.linear1 = nn.Linear(n_bands, 9 * self.endmembers)
        self.linear2 = nn.Linear(9 * self.endmembers, 6 * self.endmembers)
        self.linear3 = nn.Linear(6 * self.endmembers, 3 * self.endmembers)
        self.linear4 = nn.Linear(3 * self.endmembers, self.endmembers)
        self.bn1 = nn.BatchNorm1d(num_features=self.endmembers)
        self.soft_thresholding = DynamicalSoftThresholding([self.endmembers])
        self.linear5 = nn.Linear(self.endmembers, n_bands, bias=False)
        self.params_grid = {
            "batch_size": tune.choice([5, 10, 20, 25, 30, 100, 250]),
            "learning_rate": tune.uniform(1e-4, 1e-1),
            "weight_decay": tune.choice([1e-5, 1e-4, 1e-3, 0]),
        } if tune is not None else None
        # possible options for activation function: 
        # {"function": 'sigmoid', 'tanh', 'relu', 'leaky_relu'
        #  "param": None or float (in the case of leaky_relu)}
        self.activation_function = {
            'function': 'relu',
            'param': None
        }

    def get_endmembers(self):
        return self.linear5.weight.cpu().data.numpy()

    def forward(self, x):
        # encoder
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = self.soft_thresholding(self.bn1(x))
        abundances = sum_to_one_constraint(x, self.endmembers)
        # decoder
        x = self.linear5(abundances)
        return (abundances, x)

    def get_params_grid(self):
        """
        Returns parameters designed for this architecture for Grid Search.
        """
        return self.params_grid

    def get_activation_function(self):
        """
        Returns name of the activation function
        """
        return self.activation_function

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass
