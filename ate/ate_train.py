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
from ate.ate_loss import LossMSE

# -----------------------------------------------------------------------------


def train_once(
    net, config, data_loader, loss_function=LossMSE(), optim="adam", device="cuda:0"
):
    """
    Trains AA once with pure torch

    Parameters
    ----------
    net: autoencoder object
    config: dict with parameters
    data_loader: data loader object to train the network
    loss_type: type of loss
    optim: type of optimizer
    combined_loss: dictionary with three parameters:
                   * alpha - coefficient near MSE,
                   * beta - coefficient near SAD,
                   * gamma - coefficient near volume of simplex

    returns: trained network, final loss
    """
    if device == "cuda:0" and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if optim == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=config["learning_rate"])
    else:
        raise NotImplementedError

    # train net throughout epochs
    report_loss = 0
    net.train()
    for epoch in range(config["no_epochs"]):
        train_loss = 0
        for image, _ in data_loader:
            assert len(image.shape) == 2
            image = image.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, output = net(image)

            loss_function.update(net)
            loss = loss_function(image, output)
            loss.backward()
            ldi = loss.item()
            train_loss += ldi

            optimizer.step()

        report_loss = train_loss / len(data_loader)
        print(f"EPOCH: {epoch}, loss: {report_loss}")
    return net, report_loss


# -----------------------------------------------------------------------------


def weights_initialization(net, method="Kaiming_He_uniform"):
    """
    Prepare a proper weights initialization in DNN.

    Parameters
    ----------
    method - initialisation method

    Options:
    - kaiming_uniform: He initialization with uniform distribution (default),
    - kaiming_normal: He initialization with normal distribution,
    - xavier_uniform: Xavier (Glorot) initialization with uniform distribution,
    - xavier_normal: Xavier (Glorot) initialization with normal distribution.
    """
    if method == "Kaiming_He_uniform":
        # default PyTorch initialization
        pass

    # Xavier initialization
    # "We initialized the biases to be 0" (Glorot, Bengio)
    # He initialization
    # "We also initialize b=0 (He et al.)"
    else:
        # calculate gain in accordance to the activation function
        activation = net.get_activation_function()
        gain = torch.nn.init.calculate_gain(
            nonlinearity=activation["function"], param=activation["param"]
        )

        for module in net.modules():
            if isinstance(module, nn.Linear):
                if method == "Kaiming_He_normal":
                    # torch documentation:
                    # recommended to use only with ''relu'' or ''leaky_relu''
                    a = (
                        activation["param"]
                        if activation["function"] == "leaky_relu"
                        else 0
                    )
                    nn.init.kaiming_normal_(
                        module.weight, a=a, nonlinearity=activation["function"]
                    )
                elif method == "Xavier_Glorot_uniform":
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif method == "Xavier_Glorot_normal":
                    nn.init.xavier_normal_(module.weight, gain=gain)
                else:
                    raise NameError("This type of initialization is not implemented!")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return net


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
