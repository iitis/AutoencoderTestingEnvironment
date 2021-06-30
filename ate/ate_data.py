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

import torch
from torch.utils.data import Dataset
import numpy as np

# ----------------------------------------------------------------------------


def get_dataset(name, path, normalisation="max"):
    """
    convenience function for dataset loading

    Parameters:
    ----------
    fname: file name (numpy array w. pickle)
    normalisation: data normalisation [minmax/max/max1]

    Returns:
    --------
    Hyperspectral object
    """
    return Hyperspectral(f"{path}{name}.npz", normalisation=normalisation)


# ----------------------------------------------------------------------------


def get_serialisable_ds(dataset):
    """
    Returns a dataset that is raytune-serialisable

    Parameters:
    ----------
    dataset - hyperspectral object to be encapsulated

    Returns:
    ----------
    DS-encapsulated object
    """
    x = torch.as_tensor(dataset.X, dtype=torch.float)
    return DS(x)


# ----------------------------------------------------------------------------


class DS(Dataset):
    """
    Simplified, tune-serialisable dataset
    """

    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        return self.x[index], index

    def __len__(self):
        return len(self.x)


def spectra_normalisation(X, normalisation=None):
    """
    Spectra normalisation for flattened HSI datacubes

    Parameters:
    ----------
    X - 2D input array [examples x bands]
    normalisation - 'minmax': rescaling to the range [-0.5, 0.5]
                    'max': division by the global maximum
                    'max1': division each spectrum by their maximum value

    Returns:
    ----------
    a normalised dataset,
    a dictionary with global minimum and maximum values
    """
    global_minimum = np.min(X)
    global_maximum = np.max(X)
    values = {"minimum": global_minimum, "maximum": global_maximum}

    if normalisation == "minmax":
        return (X - global_minimum) / (global_maximum - global_minimum) - 0.5, values
    elif normalisation == "max":
        return X / global_maximum, values
    elif normalisation == "max1":
        return np.array([x / np.max(x) for x in X]), values
    else:
        assert False, "spectra_normalisation(): Bad type!"


# ----------------------------------------------------------------------------


class Hyperspectral(Dataset):
    """
    ATE Dataset class
    """

    def __init__(self, fname, normalisation=None):
        """
        dataset loading

        Parameters:
        ----------
        fname: file name (numpy array w. pickle)
        normalisation: data normalisation [minmax/max/max1]

        """
        self.record = np.load(fname, allow_pickle=True)
        self.cube = np.float32(self.record["data"])
        self.n_endmembers = len(self.record["endmembers"])
        self.endmembers = self.record["endmembers"]
        self.abundances = self.record["abundances"]
        if len(self.abundances.shape) == 3:
            self.abundances = self.abundances.reshape((-1, self.abundances.shape[2]))
        if len(self.cube.shape) == 3:
            self.cube = self.cube.reshape((-1, self.cube.shape[2]))
        self.X = np.reshape(self.cube, (-1, self.cube.shape[-1]))
        if normalisation is not None:
            # Prepare normalisation on the input image and endmembers
            self.X, values = spectra_normalisation(self.X, normalisation)
            self.maximum = values["maximum"]
            self.minimum = values["minimum"]
            self.cube = np.reshape(self.X, self.cube.shape)

    def __getitem__(self, index):
        data = self.X[index]
        return data, index

    def __len__(self):
        return len(self.X)

    def get_original(self):
        return self.cube, self.X

    def get_n_endmembers(self):
        return self.n_endmembers

    def get_abundances_gt(self):
        return self.abundances

    def get_endmembers_gt(self):
        return self.endmembers

    def get_global_maximum(self):
        return self.maximum

    def get_global_minimum(self):
        return self.minimum


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
