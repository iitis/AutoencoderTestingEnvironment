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

import os

from copy import deepcopy
from torch.utils.data import DataLoader

from ate.ate_data import get_dataset
from ate.ate_unmixing import unmix_autoencoder
from ate.ate_evaluation import evaluate_autoencoder, investigate_simplex
from ate.ate_utils import add_new_key, save_results, load_model, get_device
from ate.ate_visualise import visualise
from ate.ate_train_tune import fire_tune
from ate.ate_loss import LossMSE
from ate.ate_autoencoders import get_trained_network, get_autoencoder


# ---------------------------- Experiments -----------------------------


def experiment_simple(
    autoencoder_name,
    dataset_name,
    params_aa,
    params_global,
    loss_function=LossMSE(),
    model_path=None,
    mode="train",
    experiment_name="simple",
    n_runs=1,
):
    """
    Simple experiment: trains & returns best autoencoder after (n_runs)
    or tests a given autoencoder

    Parameters:
    ----------
        autoencoder_name: AA name, e.g. 'basic'
        dataset_name: DS name, e.g. 'Samson'
        params_aa: params needed by AA (simple exp does not use GS)
        params_global: global params
        model_path: - None (default value), model will be trained from scratch
                    - path to the file with trained model for model evaluation
                    or retraining
        mode: 'train' for training a model from scratch or retraining
               and evaluation
              'test' only for model evaluation
        experiment_name: name of the experiment (for files)
        n_runs: no. runs before best autoencoder is selected

    Returns:
    --------
        evaluation_result: dictionary with results
        abundance_image: array with abundances after unmixing
        endmembers_spectra: spectra of endmembers from AE
        ratio_inside_simplex - percentage of pixels inside simplex
        prepared_model - model with weights after training / retraining / loading
    """
    assert mode in ["train", "test"]
    filename = f"{experiment_name}_{autoencoder_name}_{dataset_name}"
    full_path = os.path.join(params_global["path_results"], filename)
    normalisation = (
        "max"
        if "normalisation" not in params_global
        else params_global["normalisation"]
    )
    dataset = get_dataset(
        dataset_name, path=params_global["path_data"], normalisation=normalisation
    )
    _, X = dataset.get_original()

    _update_dataset_info(params_global, dataset)

    # Create batches
    data_loader = DataLoader(dataset, batch_size=params_aa["batch_size"], shuffle=True)

    net = None

    # Load a model if a path is given
    if model_path is not None:
        print("Loading a model from file")
        net = load_model(
            model_path, get_autoencoder(autoencoder_name, params_aa, params_global)
        )

    # Retrain a given model or train from scratch
    if mode == "train":
        if net is not None:
            print("Model will be retrained")
        else:
            print("Model will be trained from scratch")

        net = get_trained_network(
            autoencoder_name,
            data_loader,
            params_aa=params_aa,
            params_global=params_global,
            net=net,
            n_runs=n_runs,
            loss_function=loss_function,
        )

    # Save trained / retrained / loaded model
    prepared_model = deepcopy(net)

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

    evaluation_result, output_parameters = evaluate_autoencoder(
        X,
        abundances=abundance_image,
        endmembers=endmembers_spectra,
        abundances_gt=dataset.get_abundances_gt(),
        endmembers_gt=dataset.get_endmembers_gt(),
        fname=full_path,
    )

    # Using the fact that full_path is already a path.
    # May be a problem in a future, but not now.
    abundance_image = output_parameters["sorted_abundances"]
    endmembers_spectra = output_parameters["sorted_endmembers"]
    estimated_simplex = output_parameters["reconstructed_simplex_dot_product"]
    save_results(full_path, abundance_image, endmembers_spectra)

    path_for_saving_visualisation = params_global["path_visualisations"]
    visualise(full_path, X, path_for_saving_visualisation, filename)
    _, ratio_inside_simplex = investigate_simplex(
        X, endmembers_spectra, folder=path_for_saving_visualisation, filename=filename
    )

    return (
        evaluation_result,
        abundance_image,
        endmembers_spectra,
        ratio_inside_simplex,
        prepared_model,
        estimated_simplex,
    )


# -------------------------------------------------------------------


def experiment_tune(
    autoencoder_name,
    dataset_name,
    params_aa,
    params_global,
    tune_config,
    loss_function=LossMSE(),
    experiment_name="exp_tune",
    grace=2,
    no_epochs=2,
    num_samples=1,
    resources_per_trial={"cpu": 4, "gpu": 0.25},
):
    """
    Returns best autoencoder trained with tune

    Parameters:
    ----------
    autoencoder_name: AA name, e.g. 'basic'
    dataset_name: DS name, e.g. 'Samson'
    params_aa: params needed by AA (simple exp does not use GS)
    params_global: global params
    tune_nofig: Ray Tune configuration
    experiment_name: name of the experiment (for files)
    loss_function: loss function used
    grace: grace, min. no. epochs before division (2 for speed, no_epochs for performance)
    no_epochs: max no. epochs
    resources_per_trial: no. processed are your devices divided by that

    Returns:
    --------
    best_trained_model: trained autoencoder, ready for use
    best_config: best config
    best_loss: best loss
    best_checkpoint: checkpoint of the above ae
    result: tune.analysis object, see
        https://docs.ray.io/en/master/tune/api_docs/analysis.html
    """
    normalisation = (
        "max"
        if "normalisation" not in params_global
        else params_global["normalisation"]
    )
    dataset = get_dataset(
        dataset_name, path=params_global["path_data"], normalisation=normalisation
    )
    _, X = dataset.get_original()
    _update_dataset_info(params_aa, dataset)

    device = get_device(params_global)

    best_trained_model, best_config, best_loss, best_checkpoint, result = fire_tune(
        autoencoder_name=autoencoder_name,
        loss_function=loss_function,
        grace=grace,
        no_epochs=no_epochs,
        autoencoder_params=params_aa,
        resources_per_trial=resources_per_trial,
        num_samples=num_samples,
        dataset=dataset,
        exp_name=experiment_name,
        tune_config=tune_config,
        device=device,
    )

    return best_trained_model, best_config, best_loss, best_checkpoint, result


# -------------------------------------------------------------------


def evaluate(
    trained_net,
    dataset_name,
    autoencoder_name,
    params_aa,
    params_global,
    experiment_name,
    loss_function=LossMSE(),
):
    """
    Evaluation of results

    Parameters:
    ----------
    trained_net: trained nn
    dataset_name: DS name, e.g. 'Samson'
    params_aa: params needed by AA (simple exp does not use GS)
    params_global: global params
    experiment_name: name of the experiment (for files)
    loss_function: loss function used

    Returns:
    --------
    evaluation_result: dictionary with results
    abundance_image: array with abundances after unmixing
    endmembers_spectra: spectra of endmembers from AE
    ratio_inside_simplex - percentage of pixels inside simplex
    estimated simplex - simplex
    """
    filename = f"{experiment_name}_{autoencoder_name}_{dataset_name}"
    full_path = os.path.join(params_global["path_results"], filename)

    # Save trained / retrained / loaded model
    normalisation = (
        "max"
        if "normalisation" not in params_global
        else params_global["normalisation"]
    )
    dataset = get_dataset(
        dataset_name, path=params_global["path_data"], normalisation=normalisation
    )
    _, X = dataset.get_original()
    _update_dataset_info(params_global, dataset)

    # Create batches
    data_loader = DataLoader(
        dataset, batch_size=params_aa["batch_size"], shuffle=True, num_workers=0
    )
    device = get_device(params_global)
    abundance_image, endmembers_spectra, _, _ = unmix_autoencoder(
        trained_net,
        data_loader,
        params_global["n_endmembers"],
        params_global["n_bands"],
        params_global["n_samples"],
        loss_function=loss_function,
        device=device,
    )

    evaluation_result, output_parameters = evaluate_autoencoder(
        X,
        abundances=abundance_image,
        endmembers=endmembers_spectra,
        abundances_gt=dataset.get_abundances_gt(),
        endmembers_gt=dataset.get_endmembers_gt(),
        fname=full_path,
    )

    # Using the fact that full_path is already a path.
    # May be a problem in a future, but not now.
    abundance_image = output_parameters["sorted_abundances"]
    endmembers_spectra = output_parameters["sorted_endmembers"]
    estimated_simplex = output_parameters["reconstructed_simplex_dot_product"]
    save_results(full_path, abundance_image, endmembers_spectra)

    path_for_saving_visualisation = params_global["path_visualisations"]
    visualise(full_path, X, path_for_saving_visualisation, filename)
    _, ratio_inside_simplex = investigate_simplex(
        X, endmembers_spectra, folder=path_for_saving_visualisation, filename=filename
    )

    return (
        evaluation_result,
        abundance_image,
        endmembers_spectra,
        ratio_inside_simplex,
        estimated_simplex,
    )


# -------------------------------------------------------------------


def _update_dataset_info(params_global, dataset):
    """
    Update global parameters with dataset params

    Parameters:
    ----------
    params_global: global parameters dictionary to update
    dataset: dataset from which to get parameters
    """
    _, X = dataset.get_original()
    add_new_key(params_global, "n_endmembers", dataset.get_n_endmembers())
    add_new_key(params_global, "n_bands", X.shape[1])
    add_new_key(params_global, "n_samples", X.shape[0])


# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
