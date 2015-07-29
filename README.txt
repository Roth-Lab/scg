# License

The single cell genotyper code is provided under the GPLv3 license. 
See the attached LICENSE.txt file for more information.

# Installation

The single cell genotyper is a standard Python package. It can be installed with the following command

python setup.py install

## Dependencies

- Python >= 2.7.6

- panda >= 0.16

- numpy >= 1.9.2

- scipy >= 0.15

- PyYaml

For users having difficulty installing Python or the packages required we suggest try the Miniconda (http://conda.pydata.org/miniconda.html) distribution.
In addition the `pip` package for the Python greatly aids in the installation of packages.

# Versions

## 0.3.1

- Added ability to pass initial labels for cells.

## 0.3.0

- Refactored code to be shared between models

- Added ability to input sample of origin for each cell to obtain sample specific clone proportions

- Add a work around for numerical issues (or a bug) in the doublet model

## 0.2.3

- Minor bug fix.

## 0.2.2

- Bug fix in position specific doublet model lower bound code 

## 0.2.1

- Minor bug fix in number of events reported to user at the start of run.

## 0.2.0

- Added position specific version of genotyper models

- Change CLI so that generating results files is not mandatory

- Added ability to output a file with just lower bound and convergence of a run

## 0.1.2

- Added Dirichlet mixture model

## 0.1.1

- Fixed doublet implementation to use less memory

## 0.1.0

- Initial release