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
from pathlib import Path
from scipy.io import loadmat


def process_jasper(
    dpath,
    gtpath,
    n_bands=198,
    n_rows=100,
    n_columns=100,
    n_endmembers=4,
    save=False,
    opath=Path("data/Jasper.npz"),
):
    """
    Processes Jasper Ridge dataset from https://rslab.ut.ac.ir/data (MATLAB version).

    Arguments:
    ----------
        dpath - path to data file.
        gtpath - path to gt file.
        n_bands - number of bands.
        n_rows - number of rown in data matrix.
        n_columns - number of columns in data matrix.
        n_endmembers - number of endmembers.
        save - whether to save the dataset.
        opath - path to output file.

    Returns:
    --------
        jasper_data - data matrix of shape (n_rows*n_columns, n_bands).
        jasper_M - endmembers matrix of shape (n_endmembers, n_bands).
        jasper_A - abundances matrix of shape (n_rows*n_columns, n_endmembers).
    """
    jasper_data = loadmat(dpath)["Y"]
    jasper_data = jasper_data.reshape(n_bands, n_columns, n_rows).T.reshape(-1, n_bands)
    jasper_M = loadmat(gtpath)["M"]
    jasper_M = np.swapaxes(jasper_M, 0, -1)
    jasper_A = loadmat(gtpath)["A"]
    jasper_A = jasper_A.reshape(n_endmembers, n_columns, n_rows).T.reshape(
        -1, n_endmembers
    )
    if save:
        np.savez(opath, data=jasper_data, endmembers=jasper_M, abundances=jasper_A)
    return (jasper_data, jasper_M, jasper_A)


# ------------------------------------------------------------------------

if __name__ == "__main__":
    pass
