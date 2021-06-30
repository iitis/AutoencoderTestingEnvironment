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
import numpy as np
import unittest

from util_nn import (
    sum_to_one_constraint,
    DynamicalSoftThresholding,
)

from ate.ate_evaluation import volume_simplex


class Test(unittest.TestCase):
    def test_sum_to_one_constraint(self):
        """
        Test function for sum to one constraint.
        """
        x1 = torch.from_numpy(np.array([[2, 1, 2], [0, 0, 0], [1, 0, 3], [1, 1, 1]]))
        gt1 = torch.from_numpy(
            np.array(
                [
                    [0.4, 0.2, 0.4],
                    [0.33, 0.33, 0.33],
                    [0.25, 0, 0.75],
                    [0.33, 0.33, 0.33],
                ]
            )
        )
        difference = np.absolute(gt1 - sum_to_one_constraint(x1, 3))
        self.assertLess(torch.max(difference).item(), 0.1)

    def test_dynamical_soft_thresholding(self):
        """
        Test function for dynamic soft thresholding
        """
        soft_thresholding = DynamicalSoftThresholding([3], alpha=0.2)
        x = torch.tensor(0.6).expand_as(torch.arange(3)).clone()
        result = soft_thresholding(x)
        y = torch.tensor(0.4).expand_as(torch.arange(3)).clone()
        difference = torch.abs(result - y)
        self.assertLess(torch.max(difference).item(), 0.0001)

    def test_volume_simplex(self):
        """
        Test function for the calculation of simplex volume
        """
        test_1 = torch.Tensor([[-1.0, -1.0], [5.0, -1.0], [1.0, 2.0]])
        result_1 = volume_simplex(test_1).item()
        self.assertAlmostEqual(9.0, result_1, 5)

        test_2 = torch.Tensor(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        )
        result_2 = volume_simplex(test_2).item()
        self.assertAlmostEqual(2.666667, result_2, 5)

        test_3 = torch.Tensor(
            [
                [1.0, 1.0, 1.0, -1.0 / np.sqrt(5.0)],
                [1.0, -1.0, -1.0, -1.0 / np.sqrt(5)],
                [-1.0, 1.0, -1.0, -1.0 / np.sqrt(5)],
                [-1.0, -1.0, 1.0, -1.0 / np.sqrt(5)],
                [0.0, 0.0, 0.0, 4.0 / np.sqrt(5)],
            ]
        )
        result_3 = volume_simplex(test_3).item()
        self.assertAlmostEqual(1.4907, result_3, 4)


# ------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
