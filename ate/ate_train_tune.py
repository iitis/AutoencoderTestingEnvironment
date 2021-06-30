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
import torch
from torch.utils.data import DataLoader
from torch import nn
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ate.ate_loss import LossMSE
from ate.ate_data import get_serialisable_ds
from ate.ate_autoencoders import get_autoencoder

# -------------------------------------------------------------------


def _get_optimizer(name, ae_parameters, config):
    """
    Returns parameterised optimiser

    Parameters
    ----------
    name: name in ['adam','sgd']
    ae_parameters: ae.parameters()
    config: configuration (with at least 'learning_rate' key)
    """
    if name == "adam":
        wd = config["weight_decay"] if "weight_decay" in config else 1e-5
        optimizer = torch.optim.Adam(
            ae_parameters, lr=config["learning_rate"], weight_decay=wd
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(ae_parameters, lr=float(config["learning_rate"]))
    else:
        raise NotImplementedError
    return optimizer


# -------------------------------------------------------------------


def _train_tune(
    config,
    autoencoder_name,
    autoencoder_params,
    checkpoint_dir,
    dataset,
    optim="adam",
    no_epochs=1,
    loss_function=LossMSE(),
    device="cuda:0",
):
    """
    Internal callable that creates and trains AE with tune!

    Parameters
    ----------
        config - config for tune.run
        autoencoder_name - name of the autoencoder in /architecutres
        autoencoder_params - params of the autoencoder (overridden by config)
        checkpoint_dir - checkpoint dir
        dataset - a dataset object
        optim - sgd or adam
        no_epochs - max number of training epochs
        loss_function - loss function used for training
        device: device (cpu/cuda:0)
    """
    ae = get_autoencoder(autoencoder_name, config, autoencoder_params)

    if device == "cuda:0" and torch.cuda.device_count() > 1:
        print("_train_tune: Parallel NN not supported!")

    ae.to(device).train()
    optimizer = _get_optimizer(optim, ae.parameters(), config)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True
        # num_workers=2
    )

    # train net throughout epochs
    for epoch in range(no_epochs):
        report_loss = 0
        for image, _ in data_loader:
            image = image.view(image.size(0), -1).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, output = ae(image)

            # warning, loss.update is not performed!
            loss = loss_function(image, output)
            loss.backward()
            report_loss += loss.data.item()
            optimizer.step()

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(ae.state_dict(), path + ".pt")

        report_loss = report_loss / len(data_loader)

        # Important: report to tune! (currently only loss is reported)
        tune.report(loss=report_loss)


# -------------------------------------------------------------------


def fire_tune(
    autoencoder_name,
    dataset,
    tune_config,
    autoencoder_params={},
    exp_name="",
    grace=2,
    no_epochs=2,
    num_samples=1,
    loss_function=LossMSE(),
    resources_per_trial={"cpu": 4, "gpu": 0.25},
    optim="adam",
    path_tune="tune",
    device="cuda:0",
):
    """
    Creates autoencoder trained with tune

    Parameters
    ----------
    autoencoder_name - name of the autoencoder in /architecutres
    tune_config - the searched param grid
    autoencoder_params - params of the autoencoder (overridden
        by values in tune_config)
    exp_name - name of the experiment, used to create files
        and directories during tuning
    grace - grace parameter for ray tune - how long since last
        change before another change (low for speed,
         5 is reasonable, max= no_epochs for best loss)
    no_epochs - max number of epochs
    num_samples - no. repeated experiments in a single run
    loss_function - loss function used for training
    cpus - on how many cpus to work
    gpus - on how many gpus to work
    optim - SGD or Adam
    path_tune - path to which to save checkpoints and results
    device: device ('cpu'/'cuda:0')

    returns:
    best_trained_model, best_config, best_loss, best_checkpoint,result

    """
    if tune is None:
        print("fire_tune: Tune unavailable! Returning...")
        return

    assert optim in ["adam", "sgd"]

    local_dir = os.path.join(path_tune, "checkpoints", exp_name)

    dataset = get_serialisable_ds(dataset)

    # define scheduler and reporter
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=no_epochs,
        grace_period=grace,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss"])

    part = partial(
        _train_tune,
        autoencoder_name=autoencoder_name,
        autoencoder_params=autoencoder_params,
        dataset=dataset,
        loss_function=loss_function,
        optim=optim,
        no_epochs=no_epochs,
        device=device,
    )

    # run ray tune
    result = tune.run(
        part,
        name=exp_name,
        config=tune_config,
        local_dir=local_dir,
        resources_per_trial=resources_per_trial,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=0,
        num_samples=num_samples,
        # checkpoint_score_attr="min-loss",
        fail_fast=True,
    )

    # get trial with best loss value after the last epoch
    best_trial = result.get_best_trial("loss", "min", "last")
    # best_config = result.get_best_config("loss", "min", "last")
    best_config = best_trial.config
    best_loss = best_trial.last_result["loss"]
    best_checkpoint = result.get_best_checkpoint(best_trial, "loss", "min")

    # recreate model the torch way...
    best_trained_model = get_autoencoder(
        autoencoder_name, best_config, autoencoder_params
    )
    if device == "cuda:0" and torch.cuda.device_count() > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    model_state = torch.load(f"{best_checkpoint}checkpoint.pt")
    best_trained_model.load_state_dict(model_state)

    return best_trained_model, best_config, best_loss, best_checkpoint, result


# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
