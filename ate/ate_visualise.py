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
import datetime

import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

from ate.ate_data import get_dataset

# ------------------------------------------------------------------------


def draw_simplex(
    M, col="red", ax=None, label="", marker="o", ms=40, lw=1, alpha=None, ls="--"
):
    """
    Draw endmembers simplex on 2D (as lines).
    """
    assert M.shape[1] in [2, 3]
    M = np.array(M)
    ax = plt.gca() if ax is None else ax
    a2 = alpha if alpha is not None else 1
    a1 = alpha if alpha is not None else 0.5
    if M.shape[1] == 2:
        ax.scatter(
            M[:, 0],
            M[:, 1],
            c=col,
            s=ms,
            edgecolor=col,
            alpha=a1,
            label=label,
            marker=marker,
        )
        for i, j in combinations(range(M.shape[0]), 2):
            ax.plot(
                [M[i, 0], M[j, 0]], [M[i, 1], M[j, 1]], ls, color=col, lw=lw, alpha=a2
            )
    else:
        ax.scatter(
            M[:, 0],
            M[:, 1],
            M[:, 2],
            c=col,
            s=ms,
            edgecolor=col,
            alpha=a1,
            label=label,
            marker=marker,
        )
        for i, j in combinations(range(M.shape[0]), 2):
            ax.plot(
                [M[i, 0], M[j, 0]],
                [M[i, 1], M[j, 1]],
                [M[i, 2], M[j, 2]],
                ls,
                color=col,
                lw=lw,
                alpha=a2,
            )


# -----------------------------------------------------------------------


def draw_scatter_plots(input_image, output_image, endmembers, filename):
    """
    Draws two scatter plots: with input pixels, pixels after
    reconstruction and endmembers (without any transformation)
    """
    MSE_error = mean_squared_error(input_image, output_image)
    from ate.ate_evaluation import mean_RMSE_error

    RMSE_error = mean_RMSE_error(input_image, output_image)
    # If dimensionality reduction is necessary apply PCA
    if endmembers.shape[1] not in [2, 3]:
        no_of_components = 2 if endmembers.shape[0] == 3 else 3
        pca = PCA(no_of_components, svd_solver="randomized").fit(input_image)
        input_image, output_image, endmembers = map(
            pca.transform, (input_image, output_image, endmembers)
        )
    # Prepare two scatter plots of points
    if endmembers.shape[1] == 3:
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.scatter(
            input_image[:, 0],
            input_image[:, 1],
            input_image[:, 2],
            c="grey",
            alpha=0.1,
            s=10,
        )
        ax.scatter(
            output_image[:, 0],
            output_image[:, 1],
            output_image[:, 2],
            c="blue",
            alpha=0.5,
            s=10,
        )
    else:
        plt.scatter(input_image[:, 0], input_image[:, 1], c="grey", alpha=0.5, s=10)
        plt.scatter(output_image[:, 0], output_image[:, 1], c="blue", alpha=0.5, s=10)
    # Mark endmembers and edges of simplex
    draw_simplex(endmembers, col="red")
    plt.title(
        (
            f"Input (grey) vs output pixels (blue);"
            f" MSE={MSE_error:.5f},"
            f" RMSE={RMSE_error:.5f}"
        )
    )
    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=400)
    plt.close()


# -----------------------------------------------------------------------


def visualise(fname, X, path_for_saving, experiment_name):
    """
    draw scatter plots of predicted endmembers
    """
    # Load results
    image = np.load(f"{fname}_results.npz")
    predicted_endmembers = image["endmembers"]
    predicted_abundances = image["abundances"]

    # Calculate reconstruction
    Y = np.matmul(predicted_abundances, predicted_endmembers)

    # Prepare simplex
    os.makedirs(f"{path_for_saving}", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%d%m%y_%H%M")
    saving_path = f"./{path_for_saving}/{experiment_name}_{current_time}.png"
    draw_scatter_plots(X, Y, predicted_endmembers, saving_path)


# -----------------------------------------------------------------------


def draw_points_with_simplex(
    image, endmembers, assignment, percentage, folder="./Test", filename=None
):
    """
    Draws a scatter plot and mark different colors on points inside /
    outside of simplex

    Parameters
    ----------
    image : np.array of shape (n1, n2, N) or (n1 * n2, N) where N is a dimension
        Image with input pixels to draw
    endmembers : np.array of shape (M, N)
        Vertices of (M-1)-simplex
    assignment : np.array of shape (n1 * n2,)
        Table in which value on i-th coordinate is True if i-th point
        is inside simplex, False in the opposite case
    percentage : float
        Percentage of points inside simplex in relation to the total number
        of given points.
    folder: string
        Folder in which a plot will be saved (default: './Test')
    filename: string
        Name of a file with simplex plot (default: None)
    """
    if len(image.shape) == 3:
        image = image.reshape((-1, image.shape[2]))
    # If dimensionality reduction is necessary
    if endmembers.shape[1] not in [2, 3]:
        no_of_components = 2 if endmembers.shape[0] == 3 else 3
        pca = PCA(no_of_components, svd_solver="randomized").fit(image)
        image, endmembers = map(pca.transform, (image, endmembers))
    # Prepare colors of points
    colors = np.where(assignment, "dodgerblue", "darkblue")
    # After PCA reconstruction
    if endmembers.shape[1] == 3:
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.scatter(image[:, 0], image[:, 1], image[:, 2], c=colors, alpha=0.8, s=10)
    else:
        plt.scatter(image[:, 0], image[:, 1], c=colors, alpha=0.8, s=10)
    # Mark endmembers and edges of simplex
    draw_simplex(endmembers, col="red")
    plt.title(f"{percentage:.2f}% of points is inside the simplex")

    os.makedirs(folder, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%d%m%y_%H%M")
    if filename is None:
        filename = current_time
    else:
        filename = f"{filename}_simplex_eval_{current_time}.png"

    plt.tight_layout()
    plt.savefig(f"{folder}/{filename}", dpi=400)
    plt.close()


# -----------------------------------------------------------------------


def compare_endmembers(
    dataset_name,
    estimated_endmembers,
    normalisation,
    visualisation,
    path_data,
    experiment_name,
):
    """
    Compare estimated endmembers with ground truth

    Parameters
    ----------
        dataset_name - name of the tested dataset
        estimated_endmembers - a vector with endmembers estimated by AE
    """
    # Compare endmembers
    dataset = get_dataset(dataset_name, path=path_data, normalisation=normalisation)
    endmembers_gt = dataset.get_endmembers_gt()

    plt.title("Comparison of estimated endmembers with ground truth")
    plt.scatter(
        x=estimated_endmembers[:, 0],
        y=estimated_endmembers[:, 1],
        alpha=0.85,
        label="estimated",
        color="dodgerblue",
    )
    plt.scatter(
        x=endmembers_gt[:, 0],
        y=endmembers_gt[:, 1],
        alpha=0.85,
        label="ground truth",
        color="orangered",
    )
    plt.legend()
    current_time = datetime.datetime.now().strftime("%d%m%y_%H%M")
    plt.savefig(
        (
            f"./{visualisation}/exp_"
            f"{experiment_name}_endmembers_"
            f"{current_time}.png"
        ),
        dpi=400,
    )
    plt.close()


# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
