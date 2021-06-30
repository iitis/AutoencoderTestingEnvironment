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

# -------------------------------------------------------------------


class LossFunction(object):
    """
    General loss function
    """

    def __init__(self, net=None):
        if net is not None:
            self.update(net)

    def set_name(self, name=None):
        self.name = name

    def update(self, net):
        """
        Update internal parameters based on the network architecture
        Parameters
        ----------
        net: architecture
        """
        pass

    def __call__(self, input_tensor, target_tensor):
        """
        Compute the loss function
        Parameters
        ----------
        input_tensor: 2D torch tensor
        input_tensor: 2D torch tensor
        """
        raise NotImplementedError


# -------------------------------------------------------------------


class LossMSE(LossFunction):
    """
    MSE loss
    """

    def __init__(self):
        super(LossMSE, self).set_name("MSE")

    def __call__(self, input_tensor, target_tensor):
        return nn.MSELoss(reduction="mean")(input_tensor, target_tensor)

    def get_name(self):
        return self.name


# -------------------------------------------------------------------


class LossSAD(LossFunction):
    """
    Spectral Angle Distance (SAD) loss
    """

    def __init__(self):
        super(LossSAD, self).set_name("SAD")

    def __call__(self, input_tensor, target_tensor):
        # inner product
        dot = torch.sum(input_tensor * target_tensor, dim=1)
        # norm calculations
        image = input_tensor.view(-1, input_tensor.shape[1])
        norm_original = torch.norm(image, p=2, dim=1)
        target = target_tensor.view(-1, target_tensor.shape[1])
        norm_reconstructed = torch.norm(target, p=2, dim=1)
        norm_product = (norm_original.mul(norm_reconstructed)).pow(-1)
        argument = dot.mul(norm_product)
        # for avoiding arccos(1)
        acos = torch.acos(torch.clamp(argument, -1 + 1e-7, 1 - 1e-7))
        loss = torch.mean(acos)

        if torch.isnan(loss):
            raise ValueError(
                f"Loss is NaN value. Consecutive values - dot: {dot}, \
                norm original: {norm_original}, norm reconstructed: {norm_reconstructed}, \
                    norm product: {norm_product}, argument: {argument}, acos: {acos}, \
                        loss: {loss}, input: {input_tensor}, output: {target}"
            )
        return loss

    def get_name(self):
        return self.name


# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
