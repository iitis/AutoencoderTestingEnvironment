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

import numpy as np
import torch
import unittest

from ate.ate_evaluation import (
    mean_RMSE_error,
    convert_to_numpy,
    SAD_distance,
    evaluate_autoencoder,
    investigate_simplex,
    draw_points_with_simplex,
    get_barycentric_coordinate,
    check_point_in_simplex,
    decompose,
    to_subspace,
    best_permutation,
    change_order_horizontally,
    change_order_vertically,
    check_arrays_shapes,
)


class Test(unittest.TestCase):
    def test_mean_RMSE_error(self):
        a1 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]])
        b1 = np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 2.0]])
        res_1 = mean_RMSE_error(a1, b1)
        gt_1 = 2.12132
        self.assertTrue(np.abs(res_1 - gt_1) < 0.001)

    def test_convert_to_numpy(self):
        tens_1 = torch.Tensor([1.0, 2.0, 3.0])
        array_1 = np.array([1.0, 2.0, 3.0])
        list_1 = [1, 2, 3]
        res_tens_1, res_array_1, res_list_1 = map(
            convert_to_numpy, (tens_1, array_1, list_1)
        )
        self.assertEqual("<class 'numpy.ndarray'>", str(type(res_tens_1)))
        self.assertEqual("<class 'numpy.ndarray'>", str(type(res_array_1)))
        self.assertEqual("<class 'numpy.ndarray'>", str(type(res_list_1)))

    def test_SAD_distance(self):
        a1 = np.array([[2, 1, 2]])
        b1 = np.array([[1, 4, 2]])
        res_1 = SAD_distance(a1, b1)
        self.assertTrue(np.abs(res_1 - 0.75628) < 0.001)

        a1 = np.array([[2, 1, 2], [0, 1, 0.5], [1, 0, 3], [1, 1, 1]])
        b1 = np.array([[1, 4.0, 2], [1, 0, 3], [2, 1, 5], [0, 3, 1]])
        res_1 = SAD_distance(a1, b1)
        # consecutive distances: 0.75628, 1.13265, 0.192675, 0.75204
        self.assertTrue(np.abs(res_1 - 0.708411) < 0.001)

    def test_evaluate_autoencoder(self):
        X = torch.Tensor(
            [[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 5.0, 2.5], [0.5, 0.0, 2.0, 1.0]]
        )
        abundances_gt = torch.Tensor(
            [[0.0, 0.4, 0.6], [0.8, 0.1, 0.1], [1.0, 0.0, 0.0]]
        )
        endmembers_gt = torch.Tensor(
            [[0.1, 0.5, 0.8, 1.0], [1.0, 2.0, 0.0, 3.0], [0.5, 1.0, 2.0, 3.0]]
        )
        # Change the order of rows: proper choice -> [1, 2, 0]
        endmembers = np.array(
            [[0.5, 1.0, 2.0, 3.0], [0.2, 0.7, 0.8, 1.0], [0.8, 2.2, 0.2, 2.8]]
        )
        # Change the order of columns: proper choice -> [1, 2, 0]
        abundances = np.array([[0.7, 0.1, 0.3], [0.4, 0.6, 0.0], [0.0, 1.0, 0.0]])
        result, _ = evaluate_autoencoder(
            X, abundances, endmembers, abundances_gt, endmembers_gt
        )
        print(
            f'reconstruction_error_RMSE_multiplication: {result["reconstruction_error_RMSE_multiplication"]}'
        )
        print(
            f'abundances_error_multiplication: {result["abundances_error_multiplication"]}'
        )
        print(f'endmembers_error: {result["endmembers_error"]}')
        self.assertTrue(
            np.abs(result["reconstruction_error_RMSE_multiplication"] - 1.3119) < 0.001
        )
        self.assertTrue(
            np.abs(result["abundances_error_multiplication"] - 0.10534) < 0.001
        )
        self.assertTrue(np.abs(result["endmembers_error"] - 0.08263) < 0.001)

    def test_investigate_simplex(self):
        # Custom dataset
        dataset = "./data/Custom.npz"
        points = np.load(dataset)["data"]
        gt_vertices = np.load(dataset)["endmembers"]
        own_vertices = np.array([[1.0, -1.5], [-0.5, 0.5], [0.5, 0.3]])
        vertices_list = [gt_vertices, own_vertices]
        names = ["gt", "own"]
        folder_for_saving = "./Test"
        # Only for ground truth 100% of points should be inside the simplex,
        for endmembers, name in zip(vertices_list, names):
            _, _ = investigate_simplex(
                points, endmembers, folder=folder_for_saving, filename=name
            )

        # Synthetic data
        test_vertices_1 = np.array([[-1.0, 1.0], [1.0, -2.0], [3.0, 3.0]])
        # ---- TEST 1 ------
        # all points from the following set are inside simplex
        points_test_1 = np.array([[3.0, 3.0], [1.0, 1.0], [0, 0.5], [1.2, -0.8]])
        _, no_of_points = investigate_simplex(
            points_test_1, test_vertices_1, folder=folder_for_saving, filename="test_1"
        )
        self.assertTrue(np.allclose(no_of_points, 100.0))
        # ---- TEST 2 ------
        points_test_2 = np.array(
            [[3.0, 1.0], [-1, 1], [-1.0, -1.0], [1.0, -2.0], [0.0, 2.0]]
        )
        _, no_of_points = investigate_simplex(
            points_test_2, test_vertices_1, folder=folder_for_saving, filename="test_2"
        )
        self.assertTrue(np.allclose(no_of_points, 40.0))

        # ---- TEST 3 ------
        # dimension + 1 > number of simplex vertices
        test_vertices_2 = np.array([[0.0, 0.0, 1.0], [2.0, 3.0, 0.0], [2.0, 3.0, 1.0]])
        points_test_3 = np.array(
            [
                [1.7, 51 / 20, 0.3],
                [2.0, 3.0, 1.5],
                [1.5, 3.0, 0.6],
                [1.5, 3.0, 0.0],
                [4 / 3, 2.0, 2 / 3],
            ]
        )
        assignment, no_of_points = investigate_simplex(
            points_test_3, test_vertices_2, folder=folder_for_saving, filename="test_3"
        )
        self.assertTrue(
            np.all(assignment == np.array([True, False, True, False, True]))
        )
        self.assertTrue(np.allclose(no_of_points, 60.0))

        # ---- TEST 4 ------
        # dimension + 1 > number of simplex vertices
        test_vertices_3 = np.array(
            [[0, 9, 0, 3, 3.5], [7, 8, 0, 6, 1], [4, 10, -2, 6, 6], [2, 5, -1, 3, 3]]
        )
        points_test_4 = np.array(
            [
                [1, 2, 3, 4, 5],
                [13 / 4, 8, -3 / 4, 9 / 2, 3.375],
                [3.3, 9.05, -0.95, 4.95, 4.1],
                [10, -3.5, 7, 0, 0],
                [7.1, 8.4, -1.1, 6.6, 2.95],
            ]
        )
        assignment, no_of_points = investigate_simplex(
            points_test_4, test_vertices_3, folder=folder_for_saving, filename="test_4"
        )
        self.assertTrue(
            np.all(assignment == np.array([False, True, True, False, False]))
        )
        self.assertTrue(np.allclose(no_of_points, 40.0))

    def test_get_barycentric_coordinate(self):
        # 2-simplex
        # ----- TEST 1 -------
        point = np.array([2, -1])
        vertices = np.array([[1, 2], [2, 8], [4, 3]])
        result = get_barycentric_coordinate(point, vertices)
        self.assertTrue(np.allclose(result, np.array([18 / 17, -10 / 17, 9 / 17])))

        # Example from: math.stackexchange.com
        # ----- TEST 2 -------
        point_2 = np.array([2, 1])
        vertices_2 = np.array([[-2, -1], [3, -1], [1, 4]])
        result_2 = get_barycentric_coordinate(point_2, vertices_2)
        self.assertTrue(np.allclose(result_2, np.array([0.04, 0.56, 0.4])))

        # ----- TEST 3 -------
        point_3 = np.array([2, 3])
        vertices_3 = np.array([[1, 3], [2, 0], [4, 1]])
        result_3 = get_barycentric_coordinate(point_3, vertices_3)
        self.assertTrue(np.allclose(result_3, np.array([6 / 7, -2 / 7, 3 / 7])))

        # 4-simplex
        # ---- TEST 4 -------
        point_4 = np.array([1, 2, 3, 4])
        vertices_4 = np.array(
            [[2, 5, -1, 3], [5, 3, 1, 3], [-2, 4, 1, 0], [4, 3, -1, 1], [1, 2, 1, 3]]
        )
        result_4 = get_barycentric_coordinate(point_4, vertices_4)
        self.assertTrue(
            np.allclose(
                result_4, np.array([-2 / 13, 11 / 13, 3 / 13, -11 / 13, 12 / 13])
            )
        )

    def test_check_point_in_simplex(self):
        coordinates_1 = np.array([0.1, 0.0, 1.0])
        coordinates_2 = np.array([-0.00001, 0.1, 0.3, 0.5])
        coordinates_3 = np.array([0.000001, 0.3, 0.5])
        coordinates_4 = np.array([-5, 0, 1, 3])
        self.assertTrue(check_point_in_simplex(coordinates_1))
        self.assertFalse(check_point_in_simplex(coordinates_2))
        self.assertTrue(check_point_in_simplex(coordinates_3))
        self.assertFalse(check_point_in_simplex(coordinates_4))

    def test_decompose(self):
        M = np.array(
            [[2, 5, -1, 3], [5, 3, 1, 3], [-2, 4, 1, 0], [4, 3, -1, 1], [1, 2, 1, 3]]
        )
        x0, Pv, Mp = decompose(M)
        self.assertTrue(np.all((x0 == np.array([2, 17 / 5, 1 / 5, 2]))))
        Pv_true = np.array(
            [
                [-0.940561, 0.100236, 0.123149, -0.300221],
                [-0.0935564, -0.655624, 0.664245, 0.346675],
                [0.243029, -0.420089, 0.0742667, -0.871175],
            ]
        )
        for i, v in enumerate(Pv):
            self.assertTrue(np.allclose(v, Pv_true[i]) or np.allclose(-v, Pv_true[i]))
        Mp_true = np.array(
            [
                [-0.2876222, -1.4994174, -1.63243744],
                [-3.0634792, 0.8596514, 0.08536096],
                [4.5213468, -0.1811028, 0.57759396],
                [-1.7687742, -1.0686322, 1.43614856],
                [0.5985288, 1.889501, -0.46666604],
            ]
        )
        for i, v in enumerate(Mp.T):
            self.assertTrue(
                np.allclose(v, Mp_true[:, i]) or np.allclose(-v, Mp_true[:, i])
            )

    def test_to_subspace(self):
        x0 = np.array([2, 17 / 5, 1 / 5, 2])
        Pv = np.array(
            [
                [0.940561, -0.100236, -0.123149, 0.300221],
                [-0.0935564, -0.655624, 0.664245, 0.346675],
                [0.243029, -0.420089, 0.0742667, -0.871175],
            ]
        )
        X = np.array([[1, -0.5, 0.6, 1.3], [0, 1, 1.2, 7]])
        Xp = to_subspace(x0, Pv, X)
        self.assertTrue(
            np.allclose(
                Xp,
                np.array([[-0.809055, 2.67352, 2.03485], [-0.2626, 4.15823, -3.75945]]),
            )
        )

    def test_best_permutation(self):
        xx = [[1, 5], [1, 2], [3, 0]]
        ground_truth = [[1, 2], [3, -1], [1, 3]]
        func = lambda x, y: float(np.sum(abs(x - y))) / len(xx)
        dist, perm = best_permutation(xx, ground_truth, func)
        self.assertEqual(dist, 1)
        self.assertTrue(np.all(perm == [1, 2, 0]))

    def test_change_order_horizontally(self):
        X = [[1, 5], [1, 2], [3, 0]]
        perm = [1, 2, 0]
        X_p = change_order_horizontally(X, perm)
        self.assertTrue(np.all(X_p == [[1, 2], [3, 0], [1, 5]]))

    def test_change_order_vertically(self):
        X = [[1, 5], [1, 2], [3, 0]]
        perm = np.array([1, 0])
        X_p = change_order_vertically(X, perm)
        self.assertTrue(np.all(X_p == [[5, 1], [2, 1], [0, 3]]))

    def test_check_arrays_shapes(self):
        check_arrays_shapes(
            np.array([[1, 0], [1, 2], [1, 4]]), np.array([[0, 0], [0, 1], [1, 2]])
        )
