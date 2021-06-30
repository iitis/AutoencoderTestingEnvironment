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

# -------------------------------------LOCAL RUN ------------------------------
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# -----------------------------------------------------------------------------

import numpy as np
import unittest

from ate.ate_utils import (
    assert_unique_keys,
    add_new_key,
    save_information_to_file,
    list_architectures,
)


class Test(unittest.TestCase):
    def test_list(self):
        self.assertEqual(
            list_architectures().sort(), ["basic", "original", "modified"].sort()
        )

    def test_assert_unique_keys(self):
        print("running test")
        d = {"a": 0, "b": 1}
        d1 = {"a": 0, "c": 1}
        d2 = {"c": 0, "d": 1}
        with self.assertRaises(Exception):
            assert_unique_keys(d, d1)
        try:
            assert_unique_keys(d, d2)
        except:
            self.fail()

    def test_add_new_key(self):
        d = {"a": 1}
        add_new_key(d, "a", 2)
        add_new_key(d, "b", 3)
        self.assertEqual(d, {"a": 1, "b": 3})

    def test_save_information_into_file(self):
        test_result = {"run": 3, "dataset": "Custom", "points_in_simplex": 85.31}
        keys = list(test_result.keys())
        values = list(test_result.values())
        test_path = "./Test"
        test_name = "test_file"
        full_path = f"{test_path}/{test_name}.csv"
        if os.path.isfile(full_path):
            os.remove(full_path)
        save_information_to_file(keys, values, folder=test_path, filename=test_name)

        loaded_table = np.loadtxt(
            f"{test_path}/{test_name}.csv", delimiter=";", dtype=str
        )
        loaded_keys = list(loaded_table[0])
        loaded_values = list(loaded_table[1])
        keys = [str(key) for key in keys]
        values = [str(value) for value in values]
        self.assertEqual(keys, loaded_keys)
        self.assertEqual(values, loaded_values)
