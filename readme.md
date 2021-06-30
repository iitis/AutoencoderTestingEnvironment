## DESCRIPTION:

Autoencoder Testing Environment (ATE) v.1.0
Experimental and testing environment for autoencoders

Related to the work:
> Stable training of autoencoders for hyperspectral unmixing

Source code for the review process of the 28th International Conference on Neural Information Processing (ICONIP 2021).

## LICENSE:
Copyright 2021 Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) <https://www.iitis.pl>
Authors:
- Kamil Książek (ITAI PAS, ORCID ID: [0000−0002−0201−6220](https://orcid.org/0000-0002-0201-6220)),
- Przemysław Głomb (ITAI PAS, ORCID ID: [0000−0002−0215−4674](https://orcid.org/0000-0002-0215-4674)),
- Michał Romaszewski (ITAI PAS, ORCID ID: [0000−0002−8227−929X](https://orcid.org/0000-0002-8227-929X)),
- Michał Cholewa (ITAI PAS, ORCID ID: [0000−0001−6549−1590](https://orcid.org/0000-0001-6549-1590)),
- Bartosz Grabowski (ITAI PAS, ORCID ID: [0000−0002−2364−6547](https://orcid.org/0000-0002-2364-6547)).

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

## FUNCTIONALITY:
- Autoencoder training & retraining in order to avoid bad initialisations described in the paper
- Unmixing and reconstruction error/spectra evaluation
- Simplex evaluation
- RayTune hyperparameter selection (GS+ASHA)

## RULES:

- Autoencoders architectures are loaded from <em>./architectures/</em> subfolder. One file per autoencoder.
- An example experiment using a synthetic dataset can be run from `exp_demo.py` file.

## FILES:

- `ate/ate_*.py`: Autoencoder Testing Environment core files.
- `ate/ate_tests*.py`: ATE tests.
- `exp_*.py`: Experiment files.
- `architectures/*.py`: Autoencoder files. One file contains one autoencoder.
- `util_*.py`: External libraries (to be removed in future).

## DATASETS:

All datasets have to be inserted into <em>./data/</em> folder. The example, synthetic <em>Custom</em>  is used by the demonstration experiment file (`exp_demo.py`).

## ARCHITECTURES:

Three autoencoder architectures are prepared:

- `original`: an architecture with sigmoid activation function from [Palsson et al.](https://ieeexplore.ieee.org/document/8322133) paper;
- `modified`: a modification of the above architecture;
- `basic`: a simple architecture with ReLU activation function.

## USAGE:

Copy and rename `exp_demo.py` file to `exp_<your_name_here>.py` file.
Ensure that paths in params_globals, params_aa are correct (see demo).
Put your autoencoders into <em>architectures/</em> directory with the <em>Autoencoder</em> class name.
Tests use parameters in `ate_tests.tests.params.py`, ensure that paths are consistent with paths in params.


## TUNE USAGE:

1. Copy and rename `exp_demo.py` file to `exp_<your_name_here>.py` file.
2. Put your autoencoders into <em>architectures/</em> directory with the <em>Autoencoder</em> class name.
3. Modify `run()` function's body:
    1. The only hyperparameter from `default_params_aa` that is used is the `no_epochs`, which describes for how many epochs the model will be trained after finding the best parameters.
    2. You need to define
      	- `autoencoder_name` (name that is recognizable by `get_autoencoder`, e.g. <em>'basic'</em>);
      	- `dataset_name` (name recognizable by `get_dataset`, e.g. <em>'Custom'</em>);
      	- `params_aa`, `params_global` - dictionaries with hyperparameters described in <em>'exp_demo.py</em>;
		- `tune_config`: parameters with their ranges to be found by the RayTune optimizer, details are contained in the architecture files;
		- `loss_function`: loss function for optimization, possible options: Mean Squared Error `LossMSE()` or Spectral Angle Distance `LossSAD()`;
		- `experiment_name`: anything which can be a filename;
		- `grace`: parameter of ASHA Scheduler defining interval between stopping of trials;
		- `no_epochs`: the maximum number of epochs during RayTune optimization;
		- `num_samples`: parameter defining a number of sampling from the space of hyperparameters;
		- `resources_per_trial`: CPU and GPU resources to allocate per trial.
		More details connected with RayTune parameter are contained in [the official RayTune documentation](https://docs.ray.io/en/stable/index.html).
    3. invoke `experiment_tune()` function passing arguments as described in its definition and docstring.
4. In `if __name__ == "__main__"`'s body
   1. init_env()
   2. run()
5. Run the file from shell with `python exp_<your_name>.py`.