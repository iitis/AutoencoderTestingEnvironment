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

test_params_global = {
    "path_data": "data/",  # dataset dir
    "path_results": "results",  # path of results
    "path_visualisations": "visualisations",  # path of visualisations
    "path_tune": "tune",  # path of tune
    "optim": "adam",  # optimizer (Adam by default)
    "normalisation": "max",  # a way of normalisation
    "weights_init": "Kaiming_He_uniform",  # weights initialization
    "seed": None,  # set deterministic results (or None)
}

test_params_aa = {
    "learning_rate": 0.01,
    "no_epochs": 10,
    "weight_decay": 0,
    "batch_size": 5,
}

# ---------------------------------------------------------------------------


def get_params():
    """
    Returns params for tests

    Returns:
    --------
    params_global, params_aa
    """

    return dict(test_params_global), dict(test_params_aa)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass
