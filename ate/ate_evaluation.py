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
import scipy.linalg as la
from sklearn.metrics import mean_squared_error
from itertools import permutations
from ate.ate_visualise import draw_points_with_simplex

# -------------------------------------------------------------------


def evaluate_autoencoder(
    X, abundances, endmembers, abundances_gt, endmembers_gt, fname=None
):
    """
    Evaluates autoencoder with DAEN

    Parameters:
    ----------
        X: 2D data (n_samples, n_bands)
        abundances: 2D abundance array (n_samples, n_endmembers)
        endmembers: 2D endmember array (n_endmembers, n_bands)
        abundances_gt: 2D ground truth abundance array (n_samples, n_endmembers)
        endmembers_gt: 2D ground truth endmember array (n_endmembers, n_bands)
        fname: file name for saving result (or None)

    Returns
    -------
        DAEN result,
        reconstruction output from the autoencoder (sorted endmembers, abundances
        and output)
    """
    X, abundances, endmembers, abundances_gt, endmembers_gt = map(
        convert_to_numpy, (X, abundances, endmembers, abundances_gt, endmembers_gt)
    )

    # Create results dictionaries
    keys = [
        "reconstruction_error_RMSE_multiplication",
        "reconstruction_error_MSE_multiplication",
        "abundances_error_multiplication",
        "volume_simplex_dot_product",
        "endmembers_error",
    ]
    results = dict.fromkeys(keys)

    keys = [
        "reconstructed_simplex_dot_product",
        "sorted_abundances",
        "sorted_endmembers",
    ]
    output_method = dict.fromkeys(keys)

    # Calculate endmember errors and select proper permutation
    results["endmembers_error"], order = best_permutation(
        endmembers, endmembers_gt, SAD_distance
    )
    # Prepare proper order of abundances and endmembers
    endmembers = change_order_horizontally(endmembers, order)
    abundances = change_order_vertically(abundances, order)
    output_method["sorted_endmembers"] = endmembers
    output_method["sorted_abundances"] = abundances

    # Calculate volume of simplex
    results["volume_simplex_dot_product"] = volume_simplex(
        torch.from_numpy(endmembers)
    ).item()

    # Create reconstruction of input points and calculate error values
    # for reconstruction and abundances directly from the autoencoder
    Y_multiplied = np.matmul(abundances, endmembers)
    output_method["reconstructed_simplex_dot_product"] = Y_multiplied
    results["reconstruction_error_RMSE_multiplication"] = mean_RMSE_error(
        X, Y_multiplied
    )
    results["reconstruction_error_MSE_multiplication"] = mean_squared_error(
        X, Y_multiplied
    )
    results["abundances_error_multiplication"] = mean_RMSE_error(
        abundances, abundances_gt
    )

    if fname is not None and fname != "":
        np.savez(fname, **results)

    return results, output_method


# -------------------------------------------------------------------


def SAD_distance(X, Y):
    """
    Calculate Spectral Angle Distance between X and Y

    Parameters:
    ----------
        X, Y: 2D numpy arrays with the same shapes

    Returns
    -------
        SAD value
    """
    check_arrays_shapes(X, Y)

    dot = np.sum(X * Y, axis=1)
    norm_original = np.linalg.norm(X, axis=1)
    norm_reconstructed = np.linalg.norm(Y, axis=1)
    norm_product = np.multiply(norm_original, norm_reconstructed)
    # To avoid division by 0.
    argument = np.divide(dot, norm_product + 1e-9)
    # To avoid NaNs in arccos() calculations
    np.clip(argument, -1 + 1e-9, 1 - 1e-9, out=argument)
    acos = np.arccos(argument)
    error = np.mean(acos)
    return error


# -------------------------------------------------------------------


def mean_RMSE_error(X, Y):
    """
    Calculale mean Root Mean Squared Error between X and Y arrays.

    parameters:
        X, Y: 2D numpy arrays of (M, N) shape

    Returns
    -------
        mean RMSE
    """
    check_arrays_shapes(X, Y)

    RMSE_errors = [
        mean_squared_error(X[i], Y[i], squared=False) for i in range(X.shape[0])
    ]
    return np.mean(RMSE_errors)


# -------------------------------------------------------------------


def best_permutation(xx, ground_truth, func):
    """
    Compute measure ignoring reciprocal permuations.
    Select best permutation.

    Parameters:
    ----------
        xx - calculated vectors
        ground_truth - vectors to which the distance is calculated
        func - function of distance between xx and ground_truth

    Returns
    -------
        minimal distance between xx and ground_truth
        permutation which gives the minimal distance
    """
    plist = [np.array(ii) for ii in list(permutations(range(len(xx))))]
    mlist = [func(np.array(xx)[ii], ground_truth) for ii in plist]
    return np.min(mlist), np.array(plist[np.argmin(mlist)])


# -------------------------------------------------------------------


def change_order_horizontally(X, permutation):
    """
    Change the order of the vector according to the given permutation.

    Parameters:
    ----------
        X - a vector in which the order of elements should be changed
        permutation - a new order of elements

    Returns X with the order of elements according to 'permutation'
    """
    return np.array(X)[permutation]


# -------------------------------------------------------------------


def change_order_vertically(X, permutation):
    """
    Change the order of columns in a matrix according to the permutation.

    Parameters:
    ----------
        X - a vector in which the order of columns should be changed
        permutation - a new order of columns

    Returns
    -------
    X with the order of columns according to 'permutation'
    """
    return np.array(X)[:, permutation]


# -------------------------------------------------------------------


def check_arrays_shapes(X, Y):
    """
    Check if shapes of two arrays are the same.
    In the positive case raise an error.
    """
    if X.shape != Y.shape:
        raise ValueError("Wrong vector shape!")


# -------------------------------------------------------------------


def convert_to_numpy(X):
    """
    Convert X to numpy array
    Possible options: X is torch.Tensor, numpy.ndarray or list
    """
    if type(X) == torch.Tensor:
        return X.numpy()

    elif type(X) == np.ndarray:
        return X

    elif isinstance(X, list):
        return np.asarray(X)

    else:
        raise TypeError("Wrong type of data")


# -------------------------------------------------------------------


def get_barycentric_coordinate(point, vertices):
    """
    Get coordinates of a point in a barycentric system.

    Parameters
    ----------
    point : np.array of shape (N,) where N is a dimension
        Point for which barycentric coordinates will be calculated.
    vertices : np.array of shape (M, N)
        Vertices of (M-1)-simplex

    Returns
    -------
    barycentric : np.array of shape (M,)
        Barycentric coordinates of a point
    """
    assert point.shape[0] == vertices.shape[1]
    assert vertices.shape[0] == (vertices.shape[1] + 1)
    T = np.transpose(np.subtract(vertices[:-1], vertices[-1]))
    difference = np.subtract(point, vertices[-1])
    barycentric = np.matmul(np.linalg.inv(T), difference)
    barycentric = np.append(barycentric, 1 - np.sum(barycentric))
    return barycentric


# -------------------------------------------------------------------


def check_point_in_simplex(coordinates):
    """
    Check boundary conditions of barycentric coordinates which implies
    that a given point is inside / outside of simplex.

    Parameters
    ----------
    coordinates : np.array of shape (M, )
        Barymetric coordinates of a point

    Returns
    -------
    bool
        True if a given point is inside simplex, False in the opposite case

    """
    return (coordinates >= 0.0).all()


# -------------------------------------------------------------------


def investigate_simplex(points, vertices, folder="./Test", filename=None):
    """
    Investigate which points are inside and which ones are outside the simplex
    given by vertices.

    Parameters
    ----------
    points : np.array of shape (n1 x n2 x N) or (n1 * n2 x N),
             where N is a dimension
        Table with n points which have to be checked.
    vertices : np.array of shape (M x N), where M is a number of vertices
        Table with vertices of (M-1)-simplex
    folder: string
        Folder in which a plot will be saved (default: './Test')
    filename: string
        Name of a file with simplex plot (default: None)

    Returns
    -------
    assignment : np.array of shape (n,)
        Table in which value on i-th coordinate is True if i-th point
        is inside simplex, False in the opposite case
    percentage : float
        Percentage of points inside simplex in relation to the total number
        of given points.

    """
    if len(points.shape) == 3:
        points = points.reshape((-1, points.shape[2]))
    assignment = np.zeros(points.shape[0], dtype=bool)

    # Number of vertices of simplex should be equal to N + 1 where
    # N is a dimension of consecutive points. In the ooposite case it is necessary
    # to transform points
    dimension = points.shape[1]
    no_of_vertices = vertices.shape[0]
    assert no_of_vertices <= dimension + 1
    if dimension + 1 > no_of_vertices:
        print(f"Transformation from data space to simplex space.")
        x0, Pv, vertices = decompose(vertices)
        points = to_subspace(x0, Pv, points)

    for i in range(points.shape[0]):
        barycentric_coordinate = get_barycentric_coordinate(points[i], vertices)
        check_if_inside_simplex = check_point_in_simplex(barycentric_coordinate)
        assignment[i] = check_if_inside_simplex

    number_of_points_inside_simplex = np.count_nonzero(assignment)
    percentage = 100.0 * number_of_points_inside_simplex / assignment.shape[0]
    # Prepare plot
    draw_points_with_simplex(points, vertices, assignment, percentage, folder, filename)
    return assignment, percentage


# -------------------------------------------------------------------


def decompose(M):
    """Decompose the simplex into center point, projection matrix and
    projected space coordinates: M -> x0, Pv, Mp.

    Parameters
    ----------
        M: np.array of shape (R, L) - original simplex points (row vectors).

    Returns
    -------
        x0: np.array of shape (L,) - simplex center point.
        Pv: np.array of shape (R - 1, L) - vectors for projection onto
            simplex subspace.
        Mp: np.array of shape (R, R - 1) - simplex coordinates in the
            subspace Pv
    Raises:
        (General numpy/scipy matrix algebra errors.)
    """
    x0 = np.mean(M, axis=0)
    M0 = M - x0
    U, _, _ = la.svd(M0.T, full_matrices=False)
    Pv = U[:, :-1].T
    Mp = np.dot(Pv, M0.T).T
    return x0, Pv, Mp


# ----------------------------------------------------------------------------


def to_subspace(x0, Pv, X):
    """Venture from data space to simplex space: X, Pv, x0 -> Mp."""
    return np.dot(Pv, (X - x0).T).T


# ----------------------------------------------------------------------------


def volume_simplex(vectors):
    """
    Compute volume of a simplex given as torch Tensor

    Parameters
    ----------
        vectors: simplex vectors
    Returns
    -------
        volume: simplex volume

    """
    if not (vectors.shape[0] == vectors.shape[1] + 1):
        vectors = decompose_torch(vectors)
    determinant = torch.det(torch.sub(vectors[1:], vectors[0]))
    factorial = np.math.factorial(vectors.shape[1])
    volume = torch.abs(torch.div(determinant, factorial))
    return volume


# ------------------------------------------------------------------------


def decompose_torch(matrix):
    """
    Decompose the simplex into center point, projection matrix and
    projected space coordinates. Torch-focused implementation

    Parameters
    ----------
        matrix: np.array of shape (R, L) - original simplex points (row vectors).

    Returns
    -------
        Mp: np.array of shape (R, R - 1) - simplex coordinates in the
            subspace Pv
    """

    x0 = torch.mean(matrix, dim=0)
    M0 = torch.sub(matrix, x0)
    M0 = torch.transpose(M0, 0, 1)
    # UV should be computed becasue when compute_uv=False backward
    # operations cannot be performed
    U, _, _ = torch.svd(M0, compute_uv=True)
    Pv = torch.transpose(U[:, :-1], 0, 1)
    Mp = torch.transpose(torch.matmul(Pv, M0), 0, 1)
    return Mp


# ------------------------------------------------------------------------

if __name__ == "__main__":
    pass
