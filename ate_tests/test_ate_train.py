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

# -------------------------------------LOCAL RUN ------------------------------
import sys

if __name__ == "__main__":
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    print("Run tests from ../run_ate_tests.py")
# -----------------------------------------------------------------------------

import unittest
import numpy as np
from torch.utils.data import DataLoader
from torch import nn

from ate.ate_train import train_once, weights_initialization
from ate.ate_core import _update_dataset_info
from ate.ate_autoencoders import get_autoencoder
from ate.ate_data import get_dataset
from architectures import basic
from ate_tests.test_params import get_params


class Test(unittest.TestCase):
    def test_train_once_running(self):
        """
        tests if train once runs and returns proper objects
        """
        exp_name = "demo"
        autoencoder_name = "basic"
        dataset_name = "Custom"
        params_global, params_aa = get_params()
        dataset = get_dataset(
            dataset_name, path=params_global["path_data"], normalisation="max"
        )
        data_loader = DataLoader(
            dataset, batch_size=params_aa["batch_size"], shuffle=True
        )
        _update_dataset_info(params_global, dataset)
        net = get_autoencoder(autoencoder_name, params_aa, params_global)
        optim = "adam" if "optim" not in params_global else params_global["optim"]
        net, report_loss = train_once(
            net=net, config=params_aa, data_loader=data_loader, optim=optim
        )
        self.assertIsInstance(net, basic.Autoencoder)
        self.assertEqual(net.n_bands, params_global["n_bands"])
        self.assertEqual(net.n_endmembers, params_global["n_endmembers"])

    def test_train_once_logic(self):
        """
        tests logic of train_once() by training the same network n times
        """
        autoencoder_name = "basic"
        dataset_name = "Custom"
        params_global, params_aa = get_params()
        dataset = get_dataset(
            dataset_name, path=params_global["path_data"], normalisation="max"
        )
        data_loader = DataLoader(
            dataset, batch_size=params_aa["batch_size"], shuffle=True
        )
        _update_dataset_info(params_global, dataset)
        net = get_autoencoder(autoencoder_name, params_aa, params_global)

        # every training will be just a single epoch
        # note that shuffle in dataloader is true
        params_aa["no_epochs"] = 1
        # 5 epochs of learning
        ll = []
        for i in range(5):
            net, report_loss = train_once(
                net=net,
                config=params_aa,
                data_loader=data_loader,
                optim=params_global["optim"],
            )
            ll.append(report_loss)
        # assume that the model is trained well - it is much better than the initial one
        if ll[-1] > ll[0]:
            print(f"WARNING: test_train_once_logic, {ll[-1]}>{ll[0]}")

    def test_weights_initialization(self):
        """
        tests if weights are reinitialised
        """

        def mod_crc(aa):
            vv = []
            for module in aa.modules():
                if isinstance(module, nn.Linear):
                    vv.append(np.sum(np.abs(module.weight.detach().numpy())))
            return np.asarray(vv)

        methods = [
            "Kaiming_He_uniform",
            "Kaiming_He_normal",
            "Xavier_Glorot_uniform",
            "Xavier_Glorot_normal",
        ]
        params_global, params_aa = get_params()
        params_global["n_endmembers"] = 3
        params_global["n_bands"] = 100
        params_global["n_samples"] = 1000
        net = get_autoencoder("basic", params_aa, params_global)

        crc_0 = mod_crc(net)
        self.assertEqual(len(crc_0), 3)
        for m in methods:
            net = weights_initialization(net, m)
            difference = np.min((crc_0 - mod_crc(net)) ** 2)
            if m != "Kaiming_He_uniform":
                self.assertGreater(difference, 0)
            else:
                if difference == 0:
                    print(
                        "test_weights_initialization: values should be reinitialised for 'Kaiming_He_uniform'!"
                    )
        with self.assertRaises(NameError):
            weights_initialization(net, "test_noname")
