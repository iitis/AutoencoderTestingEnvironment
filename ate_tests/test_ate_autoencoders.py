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

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    print("Run tests from ../run_ate_tests.py")

import unittest
from torch.utils.data import DataLoader

from ate_tests.test_params import get_params
from ate.ate_core import _update_dataset_info
from ate.ate_autoencoders import get_autoencoder, get_trained_network, _aa_factory
from ate.ate_data import get_dataset
from architectures import basic


class Test(unittest.TestCase):
    def setUp(self):
        self.autoencoder_name = "basic"
        self.dataset_name = "Custom"
        params_global, _ = get_params()
        self.dataset = get_dataset(
            self.dataset_name, path=params_global["path_data"], normalisation="max"
        )

    def test__aa_factory(self):
        params_global, params_aa = get_params()
        _update_dataset_info(params_global, self.dataset)
        aut = _aa_factory(basic.Autoencoder, params_aa, params_global)
        self.assertIsInstance(aut, basic.Autoencoder)
        self.assertEqual(aut.n_bands, params_global["n_bands"])
        self.assertEqual(aut.n_endmembers, params_global["n_endmembers"])

    def test_get_autoencoder(self):
        params_global, params_aa = get_params()
        _update_dataset_info(params_global, self.dataset)
        aut = get_autoencoder(self.autoencoder_name, params_aa, params_global)
        self.assertIsInstance(aut, basic.Autoencoder)
        self.assertEqual(aut.params_grid, aut.get_params_grid())
        self.assertEqual(aut.n_bands, params_global["n_bands"])
        self.assertEqual(aut.n_endmembers, params_global["n_endmembers"])

    def test_get_trained_network(self):
        params_global, params_aa = get_params()
        _update_dataset_info(params_global, self.dataset)
        data_loader = DataLoader(
            self.dataset, batch_size=params_aa["batch_size"], shuffle=True
        )
        net = get_trained_network(
            autoencoder_name=self.autoencoder_name,
            data_loader=data_loader,
            params_aa=params_aa,
            params_global=params_global,
            n_runs=2,
        )
        self.assertIsInstance(net, basic.Autoencoder)
        self.assertEqual(net.n_bands, params_global["n_bands"])
        self.assertEqual(net.n_endmembers, params_global["n_endmembers"])
