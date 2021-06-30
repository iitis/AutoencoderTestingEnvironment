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

import numpy as np
import os
import csv

from run_ate_init import init_env
from ate import experiment_simple
from ate.ate_loss import LossMSE, LossSAD
from ate.ate_data import get_dataset
from ate.ate_utils import save_information_to_file, save_model, set_seed
from ate.ate_core import get_autoencoder, _update_dataset_info
from ate.ate_train import weights_initialization
from ate.ate_evaluation import convert_to_numpy
from itertools import product

# -------------------------------------------------------------------------


def set_experiment_name(params_global, additional_name):
    """
    Prepare name of the current experiment according to
    initialized parameters in 'params_global' and an additional
    part located in 'additional_name'
    """
    weight_init = params_global["weights_init"]
    loss_type = params_global["loss_type"].get_name()
    return f"exp_{weight_init}_{loss_type}_{additional_name}"


# -------------------------------------------------------------------------


def setup_parameters():
    """
    This is an organisation of parameter setup, getting them all in one place.
    """
    params_global = {
        # paths
        "path_data": "data/",  # dataset dir
        "path_results": "results",  # path of results
        "path_visualisations": "visualisations",  # path of visualisations
        "path_tune": "tune",  # path of tune
        # optimization parameters
        "optim": "adam",  # optimizer (Adam by default)
        "normalisation": "max",  # a way of normalisation
        "seed": 5,
        "total_runs": 50,  # number of training operations per each model
        "number_of_models": 50,
    }

    params_aa = {"no_epochs": 50}

    # Build final dictionary
    p = {"global": params_global, "aa": params_aa}
    return p


# ------------------------Samson dataset - RUN FUNCTION ---------------------


def parameters_set(experiment_parameter, weights, p):
    """
    Choose a set of hyperparameters according to the given number.

    Arguments:
    ----------
    experiment_parameter - an integer value from 1 to 11
    weights - a method of weights initialization
    p - a dictionary with parameters

    Returns:
    --------
    Dictionary filled parameters according to the chosen setting
    """
    # RayTune parameters
    if experiment_parameter == 1:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossMSE()
        p["global"]["dataset_name"] = "Samson"
        p["aa"]["batch_size"] = 100
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0

    # RayTune parameters
    elif experiment_parameter == 2:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossSAD()
        p["global"]["dataset_name"] = "Samson"
        p["aa"]["batch_size"] = 100
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0

    # article parameters
    elif experiment_parameter == 3:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossSAD()
        p["global"]["dataset_name"] = "Samson"
        p["aa"]["batch_size"] = 20
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0.1

    # RayTune parameters
    elif experiment_parameter == 4:
        p["global"]["autoencoder_name"] = "basic"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossMSE()
        p["global"]["dataset_name"] = "Samson"
        p["aa"]["l_n"] = 10
        p["aa"]["batch_size"] = 4
        p["aa"]["learning_rate"] = 0.0001
        p["aa"]["weight_decay"] = 0.0

    # RayTune parameters
    elif experiment_parameter == 5:
        p["global"]["autoencoder_name"] = "basic"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossSAD()
        p["global"]["dataset_name"] = "Samson"
        p["aa"]["l_n"] = 20
        p["aa"]["batch_size"] = 4
        p["aa"]["learning_rate"] = 0.0001
        p["aa"]["weight_decay"] = 0.0

    # RayTune parameters
    elif experiment_parameter == 6:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossMSE()
        p["global"]["dataset_name"] = "Jasper"
        p["aa"]["batch_size"] = 100
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0

    # RayTune parameters
    elif experiment_parameter == 7:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossSAD()
        p["global"]["dataset_name"] = "Jasper"
        p["aa"]["batch_size"] = 100
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0

    # article parameters
    elif experiment_parameter == 8:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossMSE()
        p["global"]["dataset_name"] = "Jasper"
        p["aa"]["batch_size"] = 5
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0.1

    # article parameters
    elif experiment_parameter == 9:
        p["global"]["autoencoder_name"] = "original"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossSAD()
        p["global"]["dataset_name"] = "Jasper"
        p["aa"]["batch_size"] = 5
        p["aa"]["learning_rate"] = 0.01
        p["aa"]["weight_decay"] = 0.0
        p["aa"]["dropout"] = 0.1

    # RayTune parameters
    elif experiment_parameter == 10:
        p["global"]["autoencoder_name"] = "basic"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossMSE()
        p["global"]["dataset_name"] = "Jasper"
        p["aa"]["l_n"] = 10
        p["aa"]["batch_size"] = 20
        p["aa"]["learning_rate"] = 0.001
        p["aa"]["weight_decay"] = 0.0

    # RayTune parameters
    elif experiment_parameter == 11:
        p["global"]["autoencoder_name"] = "basic"
        p["global"]["weights_init"] = weights
        p["global"]["loss_type"] = LossSAD()
        p["global"]["dataset_name"] = "Jasper"
        p["aa"]["l_n"] = 10
        p["aa"]["batch_size"] = 4
        p["aa"]["learning_rate"] = 0.0001
        p["aa"]["weight_decay"] = 1e-05

    else:
        raise ValueError("Wrong number of experiment!")

    return p


# -------------------------------------------------------------------------


def run_mixed_algorithm(number, weight):
    """
    Main code to run mixed algorithm:
    1). Initialize N=total_runs autoencoder models.
    2). Train each model from 1). M=number_of_models times
    """
    # Prepare parameters according to given settings
    parameters = setup_parameters()
    parameters = parameters_set(number, weight, parameters)

    parameters["global"]["experiment_name"] = set_experiment_name(
        parameters["global"], additional_name=f"{number}_verification"
    )

    dataset = get_dataset(
        parameters["global"]["dataset_name"],
        path=parameters["global"]["path_data"],
        normalisation=parameters["global"]["normalisation"],
    )
    _update_dataset_info(parameters["global"], dataset)
    labels = ["experiment_name", "autoencoder_name", "dataset_name"]
    fullname = "_".join(parameters["global"][l] for l in labels)
    number_of_models = parameters["global"]["number_of_models"]
    total_runs = parameters["global"]["total_runs"]

    # Check which number of run would have the next experiment
    i, j, seed = designate_number_of_experiment(
        folder=fullname, filename=fullname, total_runs=total_runs
    )
    parameters["global"]["seed"] = seed
    # i:  counter for next model, j: counter for next run

    # To obtain deterministic results
    # Seed value will be from the range [0, number_of_models * total_runs)
    set_seed(seed)

    # ------------------------------------------------------------------
    # Create and save 'number_of_models' Autoencoder models
    if i == 0 and j == 0:
        initialize_new_models(
            parameters["global"]["number_of_models"],
            parameters["global"]["autoencoder_name"],
            parameters["aa"],
            parameters["global"],
            fullname,
        )

    print(f"Start of the experiment from {i} model and {j} run.")
    if i == number_of_models and j == 0:
        return True

    # Each model initialization will be learned 'total_runs' times
    try:
        (
            results,
            estimated_abundances,
            estimated_endmembers,
            percentage,
            trained_model,
            estimated_simplex,
        ) = experiment_simple(
            parameters["global"]["autoencoder_name"],
            parameters["global"]["dataset_name"],
            parameters["aa"],
            parameters["global"],
            loss_function=parameters["global"]["loss_type"],
            model_path=f"{fullname}/model_{i}/model_initialization",
            mode="train",
            experiment_name=parameters["global"]["experiment_name"],
            n_runs=1,
        )

        # Save trained model into file
        save_model(f"{fullname}/model_{i}/{j}_training", trained_model)
        save_output_autoencoder(
            estimated_simplex,
            estimated_abundances,
            estimated_endmembers,
            f"{fullname}/model_{i}/{j}_output",
        )

        current_run_result = {
            "model": i,
            "run": j,
            "dataset": parameters["global"]["dataset_name"],
            "no_epochs": parameters["aa"]["no_epochs"],
            "batch_size": parameters["aa"]["batch_size"],
            "learning_rate": parameters["aa"]["learning_rate"],
            "weight_decay": parameters["aa"]["weight_decay"],
            "weight_initialization": parameters["global"]["weights_init"],
            "seed": parameters["global"]["seed"],
            "loss_type": parameters["global"]["loss_type"].get_name(),
            "reconstruction_error_MSE": results[
                "reconstruction_error_MSE_multiplication"
            ],
            "reconstruction_error_RMSE": results[
                "reconstruction_error_RMSE_multiplication"
            ],
            "abundances_error": results["abundances_error_multiplication"],
            "endmembers_error": results["endmembers_error"],
            "simplex_volume": results["volume_simplex_dot_product"],
            "points_in_simplex": percentage,
        }
        if "dropout" in parameters["aa"]:
            current_run_result["dropout"] = parameters["aa"]["dropout"]

        keys = list(current_run_result.keys())
        values = list(current_run_result.values())
        save_information_to_file(keys, values, folder=fullname, filename=fullname)
        return False

    except Exception as error:
        with open("log.txt", "a+") as stream:
            stream.write(f"{i}-th model, {j}-th run \n")
            stream.write(str(error))


# ----------------------------------------------------------------------


def initialize_new_models(
    number_of_models, autoencoder_name, params_aa, params_global, path
):
    """
    Create 'number_of_models' models and save them
    into a given file
    """
    print("Initialization of new models")
    for i in range(number_of_models):
        filename = f"{path}/model_{i}/"
        os.makedirs(filename, exist_ok=True)
        net = get_autoencoder(autoencoder_name, params_aa, params_global)
        net = weights_initialization(net, params_global["weights_init"])
        save_model(f"{filename}/model_initialization", net)


# ----------------------------------------------------------------------


def save_output_autoencoder(simplex, abundances, endmembers, filename):
    """
    Save information from the trained autoencoder
    (e.g. abundances, endmembers and reconstructed simplex)
    """
    simplex, abundances, endmembers = map(
        convert_to_numpy, (simplex, abundances, endmembers)
    )

    np.savez_compressed(
        filename, endmembers=endmembers, abundances=abundances, simplex=simplex
    )


# ----------------------------------------------------------------------


def designate_number_of_experiment(folder, filename, total_runs):
    """
    Load file with experiments results and select the next model number
    and the next run number

    Arguments:
    ----------
        folder - a location of the file
        filename - name of the .csv file with results
        total_runs - number of runs per each model

    Returns:
    --------
        next model number, next run number, number of lines
        in the results file
    """
    path = f"{folder}/{filename}.csv"
    if os.path.isfile(path):
        results_file = open(path)
        lines = len(list(csv.reader(results_file))) - 1  # without header
        results_tuple = divmod(lines, total_runs)
        # model and runs numeration starts from 0
        number_of_next_model = results_tuple[0]
        number_of_next_run = results_tuple[1]
        return (number_of_next_model, number_of_next_run, lines)

    else:
        # if file does not exist
        return (0, 0, 0)


# ----------------------------------------------------------------------


def test_designate_number_of_experiment():
    """
    Unittest of function designate_number_of_experiment()
    """

    def append_row_to_file(filename, elements):
        with open(filename, "a+") as stream:
            np.savetxt(stream, np.array(elements)[np.newaxis], delimiter=";", fmt="%s")

    os.makedirs("./Test", exist_ok=True)
    os.chdir("./Test")

    # ------------ TEST 1 -------------------------
    test_filename_1 = "test_next_experiment_number_1"
    if os.path.exists(f"{test_filename_1}.csv"):
        os.remove(f"{test_filename_1}.csv")
    test_file_1 = ["header", "0;0", "0;1", "0;2", "1;0"]
    for element in test_file_1:
        append_row_to_file(f"{test_filename_1}.csv", element)

    assert (1, 1, 4) == designate_number_of_experiment(os.getcwd(), test_filename_1, 3)

    # ------------ TEST 2 -------------------------
    test_filename_2 = "test_next_experiment_number_2"
    if os.path.exists(f"{test_filename_2}.csv"):
        os.remove(f"{test_filename_2}.csv")
    test_file_2 = ["header", "0;0", "0;1", "0;2", "1;0", "1;1", "1;2"]
    for element in test_file_2:
        append_row_to_file(f"{test_filename_2}.csv", element)

    assert (2, 0, 6) == designate_number_of_experiment(os.getcwd(), test_filename_2, 3)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    init_env()
    methods = [
        "Kaiming_He_uniform",
        "Kaiming_He_normal",
        "Xavier_Glorot_uniform",
        "Xavier_Glorot_normal",
    ]
    # Select the number of experiments which will be proceed
    experiments_number = [i + 1 for i in range(10, 11)]
    sets = list(product(methods, experiments_number))
    results_state = np.full(len(sets), False)

    done = False
    while not done:
        for num_of_parameters in range(len(sets)):
            weight_method = sets[num_of_parameters][0]
            number = sets[num_of_parameters][1]
            results_state[num_of_parameters] = run_mixed_algorithm(
                number=number, weight=weight_method
            )
            # to check whether all experiments are already performed
            if np.sum(results_state) == len(sets):
                done = True
                break
