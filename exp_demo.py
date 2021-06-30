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

from ate import experiment_simple
from run_ate_init import init_env
from ate.ate_loss import LossMSE
from ate.ate_visualise import compare_endmembers

# ---------------------------- PARAMETRISATION----------------------------------

# Change this name in accordance with file name!
EXP_ENVIRONMENT_NAME = "demo"

params_global = {
    "path_data": "data/",  # dataset dir
    "path_results": "results",  # path of results
    "path_visualisations": "visualisations",  # path of visualisations
    "path_tune": "tune",  # path of tune
    "optim": "adam",  # optimizer (Adam by default)
    "normalisation": "max",  # a way of normalisation
    "weights_init": "Kaiming_He_uniform",  # weights initialization
    "seed": None,  # set deterministic results (or None)
}

default_params_aa = {
    "learning_rate": 0.01,
    "no_epochs": 10,
    "weight_decay": 0,
    "batch_size": 5,
}

# ---------------------------- RUN FUNCTION ------------------------------------


def run():
    """
    Main code to run
    """
    autoencoder_name = "basic"
    dataset_name = "Custom"
    experiment_name = "demo"
    n_runs = 1
    params_aa = default_params_aa

    print(f"Global parameters: {params_global}.")
    print(f"Autoencoder parameters: {params_aa}.")
    (
        evaluation_result,
        abundance_image,
        endmembers_spectra,
        percentage,
        _,
        _,
    ) = experiment_simple(
        autoencoder_name,
        dataset_name,
        params_aa,
        params_global,
        experiment_name=experiment_name,
        mode="train",
        n_runs=n_runs,
        loss_function=LossMSE(),
    )
    compare_endmembers(
        dataset_name,
        endmembers_spectra,
        params_global["normalisation"],
        params_global["path_visualisations"],
        params_global["path_data"],
        EXP_ENVIRONMENT_NAME,
    )
    print("Results:", evaluation_result)


# -------------------------- RUN DEMO -------------------------------------

if __name__ == "__main__":
    init_env()
    run()
