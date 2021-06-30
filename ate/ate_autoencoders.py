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

import importlib
import inspect
import warnings

import numpy as np

from ate.ate_train import train_once, weights_initialization
from ate.ate_data import get_dataset
from ate.ate_evaluation import evaluate_autoencoder, investigate_simplex
from ate.ate_utils import (
    assert_unique_keys,
    add_new_key,
    save_results,
    load_model,
    get_device,
)
from ate.ate_loss import LossMSE

# ----------------------------------- Object generators ----------------


def get_trained_network(
    autoencoder_name,
    data_loader,
    params_aa={},
    params_global={},
    loss_function=LossMSE(),
    net=None,
    n_runs=1,
):
    """
    Returns best trained autoencoder object from n_runs trials (fires the factory)

    Parameters:
    ----------
    autoencoder_name: Autoencoder name
    data_loader: the instance of data loader
    params_aa: autoencoder param dic
    params_global: global dic
    loss_function: loss function
    net: and instance of autoencoder (for retraining)
    n_runs: no retrainings

    Returns:
    ----------
    trained network

    """

    device = get_device(params_global)

    optim = "adam" if "optim" not in params_global else params_global["optim"]
    nets = []
    if net is not None:
        n_runs = 1
        warnings.warn("retraining, n_runs set to 1")

    for n in range(n_runs):
        if net is None:
            net = get_autoencoder(autoencoder_name, params_aa, params_global)
            net = weights_initialization(net, params_global["weights_init"])
        net, loss = train_once(
            net=net,
            config=params_aa,
            data_loader=data_loader,
            loss_function=loss_function,
            optim=optim,
            device=device,
        )
        nets.append((net, loss))
        net = None

    arg = np.argmax([n[1] for n in nets])
    net = nets[arg][0]
    return net


def get_autoencoder(autoencoder_name, params, *args):
    """
    returns (untrained) autoencoder object from (name,params) (fires the factory)

    Parameters:
    ----------
    autoencoder_name: name of the autoencoder
    params, *args: at least one param dictionary

    Returns:
    ----------
    neural network
    """
    aa = importlib.import_module(f"architectures.{autoencoder_name}")
    try:
        aa_class_name = aa.AA_CLASS_NAME
    except AttributeError:
        aa_class_name = "Autoencoder"
    aut = _aa_factory(getattr(aa, aa_class_name), params, *args)
    return aut


# ------------------------------------ Internals ----------------------------------------------------


def _aa_factory(aa_class, params, *args):
    """
    Autoencoder factory: creates autoencoder object

    Parameters:
    ----------
    aa_class: Autoencoder class
    params_aa: autoencoder param dic
    params_global: global dic

    Returns:
    ----------
    neural network
    """

    for dd in args:
        assert isinstance(dd, dict)
    sig = inspect.signature(aa_class.__init__)
    pdic = {}
    for k, v in sig.parameters.items():
        if k == "self":
            continue
        if k in params:
            pdic[k] = params[k]
        elif True in [k in dd for dd in args]:
            for dd in args:
                if k in dd:
                    pdic[k] = dd[k]
            try:
                pdic[k]
            except:
                raise NameError("_aa_factory")
        elif v.default != inspect._empty:
            pdic[k] = v.default
        else:
            raise TypeError(f"parameter '{k}'' not defined for {aa_class}")
    return aa_class(**pdic)


# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
