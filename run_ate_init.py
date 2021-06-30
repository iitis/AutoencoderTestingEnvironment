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

import sys
import os


# Change this name in accordance with file name!
EXP_ENVIRONMENT_NAME = "demo"

# ------------------------------------------------------------------------


def ensure_dir(*args):
    """
    makes sure all needed directories exist

    *args - list of paths to check
    """
    for arg in args:
        if not os.path.isdir(arg):
            os.mkdir(arg)


# ------------------------------------------------------------------------


def init_env():
    ensure_dir(
        "results",
        "tune",
        os.path.join("tune", "visualisations"),
        os.path.join("tune", "checkpoints"),
        os.path.join("tune", "min_losses"),
        *sys.argv[1:]
    )


# ------------------------------------------------------------------------

if __name__ == "__main__":
    init_env()
