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
from torch import nn
import numpy as np
from torch import from_numpy

from ate_tests.test_params import get_params
from ate.ate_loss import LossMSE, LossSAD


def spectral_angle(a, b):
    """
    spectral angle
    """
    va = a / np.sqrt(a.dot(a))
    vb = b / np.sqrt(b.dot(b))
    return np.arccos(np.clip(va.dot(vb), -1, 1))


def msad(X, Y):
    """
    mean spectral angle
    """
    assert len(X) == len(Y)
    return np.mean([spectral_angle(X[i], Y[i]) for i in range(len(X))])


class Test(unittest.TestCase):
    def test_loss_function_sanity(self):
        for f in [LossMSE(), LossSAD()]:
            a = from_numpy(np.random.rand(10, 2))
            b = from_numpy(np.random.rand(10, 2))
            res = f(a, b).numpy()
            self.assertGreater(res, 0)
            if res != 0:
                print(f"test_loss_function_sanity: {res}!=0 for {f}")

    def test_loss_function(self):
        for i in range(10):
            A = np.random.rand(10, 2)
            B = np.random.rand(10, 2)
            if i > 7:
                A *= 0.001
                B *= 0.001
            a = msad(A, B)
            b = LossSAD()(from_numpy(A), from_numpy(B)).numpy()
            if np.abs(a - b) > 0.01:
                print("test_loss_function: SAD, large error for near-zero values!")
            a = nn.MSELoss(reduction="mean")(from_numpy(A), from_numpy(B)).numpy()
            b = LossMSE()(from_numpy(A), from_numpy(B)).numpy()
            self.assertAlmostEqual(a, b, 3)
