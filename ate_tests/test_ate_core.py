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

import os
import unittest
from torch.utils.data import DataLoader

from ate.ate_core import experiment_simple, _update_dataset_info, _update_dataset_info
from ate.ate_autoencoders import get_autoencoder, get_trained_network, _aa_factory
from ate.ate_evaluation import evaluate_autoencoder
from ate.ate_data import get_dataset
from ate.ate_unmixing import unmix_autoencoder
from ate.ate_loss import LossMSE
from ate.ate_visualise import visualise
from ate.ate_utils import get_device, save_results
from architectures import basic
from ate_tests.test_params import get_params


class Test(unittest.TestCase):
    aa_names = ["basic", "original", "modified"]

    def setUp(self):
        self.exp_name = "demo"
        self.autoencoder_name = "basic"
        self.dataset_name = "Custom"
        # params_global must be created separately for each test,
        # because some modules may modify it and affect other tests.
        params_global, _ = get_params()
        self.dataset = get_dataset(
            self.dataset_name, path=params_global["path_data"], normalisation="max"
        )

    def test_autoencoders(self):
        """
        autoencoder test: raw and temporary
        """
        params_global, _ = get_params()
        for autoencoder_name in self.aa_names:
            print(autoencoder_name)
            self.assertTrue(
                check_autoencoder(autoencoder_name, params_global=params_global)
            )

    def test_experiment_simple(self):
        """
        tests if train once runs and returns proper objects
        """
        params_global, params_aa = get_params()
        results, abundance_image, endmembers, percentage, _, _ = experiment_simple(
            autoencoder_name=self.autoencoder_name,
            dataset_name=self.dataset_name,
            params_aa=params_aa,
            params_global=params_global,
            experiment_name="unittest",
            n_runs=1,
        )
        self.assertEqual(abundance_image.shape[0], params_global["n_samples"])
        self.assertEqual(abundance_image.shape[1], params_global["n_endmembers"])
        self.assertEqual(endmembers.shape[0], params_global["n_endmembers"])
        self.assertEqual(endmembers.shape[1], params_global["n_bands"])

    def test__update_dataset_info(self):
        params_global, _ = get_params()
        params_global_copy = dict(params_global)
        _update_dataset_info(params_global, self.dataset)
        for key in params_global_copy:
            self.assertEqual(params_global_copy[key], params_global[key])
        _, X = self.dataset.get_original()
        self.assertEqual(params_global["n_endmembers"], self.dataset.get_n_endmembers())
        self.assertEqual(params_global["n_bands"], X.shape[1])
        self.assertEqual(params_global["n_samples"], X.shape[0])


# ---------------------------------------------------------------------------------


def check_autoencoder(
    autoencoder_name, params_global, loss_function=LossMSE(), dataset_name="Custom"
):
    """
    tests if autoencoder works
    """
    params_aa = {
        "learning_rate": 0.01,
        "no_epochs": 2,
        "weight_decay": 1e-5,
        "batch_size": 5,
    }

    dataset = get_dataset(
        dataset_name, path=params_global["path_data"], normalisation="max"
    )
    _, X = dataset.get_original()

    # temporary update of global dictionary
    params_global["n_endmembers"] = dataset.get_n_endmembers()
    params_global["n_bands"] = X.shape[1]
    params_global["n_samples"] = X.shape[0]
    params_global["seed"] = 3

    # create batches
    data_loader = DataLoader(
        dataset,
        batch_size=params_aa["batch_size"],
        shuffle=True,
        drop_last=True
        # num_workers=0
    )

    net = get_trained_network(
        autoencoder_name,
        data_loader,
        params_aa=params_aa,
        params_global=params_global,
        loss_function=loss_function,
    )

    device = get_device(params_global)

    abundance_image, endmembers_spectra, _, _ = unmix_autoencoder(
        net,
        data_loader,
        params_global["n_endmembers"],
        params_global["n_bands"],
        params_global["n_samples"],
        loss_function=loss_function,
        device=device,
    )
    evaluation_result, _ = evaluate_autoencoder(
        X,
        abundances=abundance_image,
        endmembers=endmembers_spectra,
        abundances_gt=dataset.get_abundances_gt(),
        endmembers_gt=dataset.get_endmembers_gt(),
    )
    # Save results and visualisations
    folder_name = "./Test"
    os.makedirs(folder_name, exist_ok=True)
    path = f"{folder_name}/{autoencoder_name}"
    save_results(path, abundance_image, endmembers_spectra)
    visualise(path, X, folder_name, autoencoder_name)

    if evaluation_result["reconstruction_error_RMSE_multiplication"] >= 0:
        return True
    return False
