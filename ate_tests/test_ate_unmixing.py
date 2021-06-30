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
from torch.utils.data import DataLoader

from ate.ate_unmixing import unmix_autoencoder
from ate.ate_loss import LossMSE
from ate.ate_train import train_once
from ate.ate_core import _update_dataset_info
from ate.ate_autoencoders import get_autoencoder
from ate.ate_data import get_dataset
from ate_tests.test_params import get_params


class Test(unittest.TestCase):
    def test_unmix_autoencoder(self):
        autoencoder_name = "basic"
        dataset_name = "Custom"
        params_global, params_aa = get_params()
        dataset = get_dataset(
            name=dataset_name, path=params_global["path_data"], normalisation="max"
        )
        data_loader = DataLoader(
            dataset=dataset, batch_size=params_aa["batch_size"], shuffle=True
        )
        _update_dataset_info(params_global, dataset)
        net = get_autoencoder(autoencoder_name, params_aa, params_global)
        optim = "adam" if "optim" not in params_global else params_global["optim"]
        net, _ = train_once(
            net=net, config=params_aa, data_loader=data_loader, optim=optim
        )
        abundance_image, endmembers_spectra, test_image, total_loss = unmix_autoencoder(
            model=net,
            data_loader=data_loader,
            n_endmembers=params_global["n_endmembers"],
            n_bands=params_global["n_bands"],
            n_samples=params_global["n_samples"],
            loss_function=LossMSE(),
            device="cpu",
        )
        self.assertIsInstance(total_loss, float)
        self.assertEqual(
            abundance_image.shape,
            (params_global["n_samples"], params_global["n_endmembers"]),
        )
        self.assertEqual(
            endmembers_spectra.shape,
            (params_global["n_endmembers"], params_global["n_bands"]),
        )
        self.assertEqual(
            test_image.shape, (params_global["n_samples"], params_global["n_bands"])
        )
