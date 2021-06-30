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

import scipy.stats as stats
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt

from itertools import product

# ----------------------------------------------------------------------


def select_data(number_of_models, number_of_runs, models_idx, table, variable):
    """
    Select data from pandas DataSeries for further experiments

    Arguments:
    ----------
    number_of_models - an integer which represents the total
                       number of models
    number_of_runs - an integer which represents the total
                     number of runs for each model
                     (this value should be te same for all models)
    models_idx - numpy ndarray which has a collection of all models
                 (for instance [1, 2, 3, 10, 15])
    table - pandas DataFrame with all results
    variable - a variable which will be extracted from table
               (for instance reconstruction error)

    Returns:
    --------
    results - numpy ndarray of shape [number_of_models, number_of_runs]
              with data samples
    """
    results = np.zeros((number_of_models, number_of_runs))

    for i in range(number_of_models):
        model = models_idx[i]
        this_model_sample = (table.loc[table["model"] == model][variable]).to_numpy()
        results[i] = np.copy(this_model_sample)

    return results


# ---------------------------------------------------------------------


def check_variance_equality(data):
    """
    Check the equality of variance using Levene's test

    Argument:
    ---------
    data - numpy ndarray with data samples
    """
    statistic, p_value = stats.levene(*data)
    if p_value < 0.05:
        print(f"WARNING! p-value is less than 0.05: {p_value}.")
    return statistic, p_value


# ---------------------------------------------------------------------


def prepare_post_hoc_test(data, number, weight_method, day):
    """
    Prepare post-hoc Conover-Iman test

    Arguments:
    ----------
    data - numpy ndarray with data samples
    number - an unique ID for given experiment
    weight_method - an abbreviation for the weight method description
    day - the day of the experiment prepairing
    """
    post_hoc_test = sp.posthoc_conover(data)
    # diagonal, NS, p < 0.001, p < 0.01, p < 0.05
    cmap = ["white", "blanchedalmond", "mediumblue", "royalblue", "lightsteelblue"]
    heatmap_args = {
        "linewidths": 0.25,
        "linecolor": "0.5",
        "cmap": cmap,
        "clip_on": False,
        "square": True,
        "cbar_ax_bbox": [0.83, 0.35, 0.04, 0.3],
    }
    post_hoc_test.to_csv(
        f"{number}_{weight_method}_{day}_post_hoc_test_table.csv",
        index=False,
        header=False,
    )
    sp.sign_plot(post_hoc_test, **heatmap_args)
    plt.title(
        f"Post-hoc analysis of Conover-Iman test for {number}_{weight_method}_{day}",
        fontdict={"horizontalalignment": "right"},
        position=(3.0, 1.85),
    )
    plt.savefig(f"{number}_{weight_method}_{day}_post_hoc_test.pdf", dpi=300)
    plt.close()


# ----------------------------------------------------------------------


def prepare_statistical_tests(
    experiment_ids, weights_methods, experiment_days, experiment_data, variable
):
    """
    Prepare ANOVA and Kruskal-Wallis analysis for all files in a given
    location according to file patterns

    Arguments:
    ----------
    experiment_ids - list with IDs of all experiments for which
                     statistical tests have to be performed
    weights_methods - list with abbreviations of weights methods
                      for which statistical tests have to be performed
    experiment_days - list with all considered days of experiments
    experiment_data - details about experiments
    variable - a variable which will be extracted from results
               (for instance reconstruction error)

    Returns:
    --------
    pandas DataFrame with following columns:
    [ID of the experiment, weight method, AE architecture, dataset,
    loss, F-value, p-value for ANOVA, H-value, p-value for Kruskal]
    """
    test_results_for_experiments = []
    for weight_method, number, day in product(
        weights_methods, experiments_ids, experiment_days
    ):
        filename = f"{number}_{weight_method}_{day}_full_results.csv"
        if os.path.isfile(filename):
            # Load current file into pandas Series
            print(f"Preparing {filename}")
            table = pd.read_csv(f"{os.getcwd()}/{filename}", delimiter=";")
            architecture = experiment_data[number]["architecture"]
            dataset = experiment_data[number]["dataset"]
            loss = experiment_data[number]["loss"]

            # count number of models and runs per each model
            models_idx = table["model"].unique()
            number_of_models = len(models_idx)
            number_of_runs = len(table["run"].unique())
            # create an empty numpy array for samples
            results = select_data(
                number_of_models, number_of_runs, models_idx, table, variable
            )

            # Check one of the assumptions for ANOVA test
            W, Leven_pvalue = check_variance_equality(results)
            # Kruskal-Wallis analysis
            Hvalue, kruskal_pvalue = stats.kruskal(*results)
            # post-hoc Conover-Iman test results
            prepare_post_hoc_test(results, number, weight_method, day)

            tests_single_result = {
                "experiment_id": number,
                "weight_method": weight_method,
                "architecture": architecture,
                "dataset": dataset,
                "loss": loss,
                "W-value": W,
                "p-value_Leven": Leven_pvalue,
                "H-value": Hvalue,
                "p-value_Kruskal": kruskal_pvalue,
            }
            test_results_for_experiments.append(tests_single_result)
    test_results_for_experiments = pd.DataFrame(test_results_for_experiments)
    return test_results_for_experiments


# -------  UNITTESTS   ---------------------------------------------


def unittest_Kruskal_Wallis_Conover():
    """
    Examples presented in:
    S. Washington et al. 'Scientific approaches to transportation research'
    """
    results = np.array(
        [
            [7, 7, 15, 11, 9],
            [12, 17, 12, 18, 18],
            [14, 18, 18, 19, 19],
            [19, 25, 22, 19, 23],
            [7, 10, 11, 15, 11],
        ]
    )
    # Unittest for Kruskal-Wallis test
    fvalue, pvalue = stats.kruskal(*results)
    assert np.abs(fvalue - 19.063658) < 0.001
    # chi-squared distribution with 4 degrees of freedom
    assert np.abs(pvalue - 0.000764) < 0.00001

    # Unittest for Conover-Iman test
    post_hoc_pvalues = sp.posthoc_conover(results).to_numpy()
    # Results:
    # (1, 2), (1, 3), (1, 4), (1, 5)
    # t-values: [3.34909, 5.00189, 7.48109, 0.52194]
    # p-values: [0.003195, 0.00006843, 0.0000003233, 0.6074]
    # (2, 3), (2, 4), (2, 5)
    # t-values: [1.65280, 4.08850, 2.82716]
    # p-values: [0.114, 0.000572, 0.01041]
    # (3, 4), (3, 5)
    # t-values: [2.43570, 4.47996]
    # p-values: [0.02434, 0.0002292]
    # (4, 5)
    # t-values: [6.91566]
    # p-values: [0.000001024]
    p_values_gt = np.array(
        [
            [1.0, 0.003195, 0.00006843, 0.0000003233, 0.6074],
            [0.003195, 1.0, 0.114, 0.000572, 0.01041],
            [0.00006843, 0.114, 1.0, 0.02434, 0.0002292],
            [0.0000003233, 0.000572, 0.02434, 1.0, 0.000001024],
            [0.6074, 0.01041, 0.0002292, 0.000001024, 1.0],
        ]
    )
    difference = np.abs(post_hoc_pvalues - p_values_gt)
    assert np.max(difference) < 0.0001


# ----------------------------------------------------------------------


def unittest_Levene():
    """
    Unittests for Levene's test of equality of variances.
    """
    data = np.array([[1, 3, 4, 7], [8, 2, 0, 1], [1, 1, 0, 9]])
    # test statistic has F distribution with (2, 9) degrees of freedom
    # 1)
    statistic, p_value = stats.levene(*data, center="median")
    assert np.abs(statistic - 0.039346) < 0.001
    assert np.abs(p_value - 0.9616) < 0.001
    # 2)
    statistic, p_value = stats.levene(*data, center="mean")
    assert np.abs(statistic - 0.555788) < 0.001
    assert np.abs(p_value - 0.5921) < 0.001


# ----------------------------------------------------------------------


def unittest_select_data():
    """
    Unittest of function select_data()
    """
    test_table = np.array(
        [
            [0, 0, 0.341],
            [0, 1, 1.241],
            [0, 2, -1.3587],
            [10, 0, 5.3789],
            [10, 1, 2.8794],
            [10, 2, 4.7954],
        ]
    )
    test_dataframe = pd.DataFrame(test_table, columns=["model", "run", "value"])
    models_idx = test_dataframe["model"].unique()
    proper_models_idx = np.array([0, 10])
    assert (models_idx == proper_models_idx).all()
    proper_result = np.array([[0.341, 1.241, -1.3587], [5.3789, 2.8794, 4.7954]])
    result = select_data(
        number_of_models=2,
        number_of_runs=3,
        models_idx=models_idx,
        table=test_dataframe,
        variable="value",
    )
    assert (proper_result == result).all()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    unittest_Kruskal_Wallis_Conover()
    unittest_Levene()
    unittest_select_data()

    experiments_ids = [
        "F001",
        "F002",
        "F003",
        "F004",
        "F005",
        "F006",
        "F007",
        "F008",
        "F009",
        "F010",
        "F011",
    ]
    weights_methods = ["XGU", "XGN", "KHN", "KHU"]
    experiment_days = ["XXXX2021"]
    experiment_data = {
        "F001": {"architecture": "original", "dataset": "Samson", "loss": "MSE"},
        "F002": {"architecture": "original", "dataset": "Samson", "loss": "SAD"},
        "F003": {"architecture": "original", "dataset": "Samson", "loss": "SAD"},
        "F004": {"architecture": "basic", "dataset": "Samson", "loss": "MSE"},
        "F005": {"architecture": "basic", "dataset": "Samson", "loss": "SAD"},
        "F006": {"architecture": "original", "dataset": "Jasper", "loss": "MSE"},
        "F007": {"architecture": "original", "dataset": "Jasper", "loss": "SAD"},
        "F008": {"architecture": "original", "dataset": "Jasper", "loss": "MSE"},
        "F009": {"architecture": "original", "dataset": "Jasper", "loss": "SAD"},
        "F010": {"architecture": "basic", "dataset": "Jasper", "loss": "MSE"},
        "F011": {"architecture": "basic", "dataset": "Jasper", "loss": "SAD"},
    }

    variable = "reconstruction_error_RMSE"
    # Apply statistical tests
    results = prepare_statistical_tests(
        experiments_ids, weights_methods, experiment_days, experiment_data, variable
    )
    # Save Pandas dataframe as .csv file
    results.to_csv("summary_results_statistics.csv", index=False)
