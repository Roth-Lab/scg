# License

The single cell genotyper code is provided under the GPLv3 license.
See the attached LICENSE.txt file for more information.

# Installation

The quickest way to install the scg software is using conda.
The following command will install the software using the bioconda repository.

```
conda install scg -c bioconda -c conda-forge
```

## Manual install

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

# Usage

The following will command will run the scg software using the examples files found in the examples/ directory of the github repo.
This assumes you have checked out the repository and are running the command from inside the root folder of the repository.

To run the doublet aware model.
Note you must create the output directory first.

```
mkdir doublet
scg run_doublet_model --config_file examples/config.yaml --state_map_file examples/state_map.yaml --out_dir doublet
```

To run without doublet handling the use the following comman.

```
mkdir no_doublet
scg run_singlet_model --config_file examples/config.yaml --out_dir no_doublet/
```

## Input Files

Several input files are required to use the scg software.

### Config File

The configuration file is specified using the ``--config_file` flag.
The configuration file specifies the parameters for the model and path to the input data files(s).
The configuration file is in YAML format with the following fields.

```
num_clusters: 40 # Number of clusters. Set to a number higher than expected as not all will be used.

alpha_prior: [9, 1] # Beta for doublet probability. Only used by doublet model.

kappa_prior: 1 # Symmetric Dirichlet prior for cluster proportions

# This section specifies the path to the data and the data specific parameters
data:
  snv:
    file: examples/snv.tsv.gz # Path to file with SNV data.

    # Dirichlet prior for the "emmission density". For SNVs there are three states: AA, AB, BB which correspond to rows.
    # Each row can be controls how likely we are to observe an event of a type given the true type.
    # The priors below assume AA and BB are reliably observed when that is the true genotypes.
    # For AB we assume there is a high probability of observing AB but also reasonable probability of AA or BB due to dropout
    gamma_prior: [[98, 1, 1], # AA
                  [25, 50, 25], # AB
                  [1, 1, 98]] # BB

    # Dirichlet prior for true state being AA, AB, BB
    state_prior: [1, 1, 1]

  # Dirichlet prior for breakpoint data, or any other type of binary character.
  breakpoint:
    file: examples/breakpoint.tsv.gz

    # For breakpoints there are only to states present/absent.
    gamma_prior: [[9, 1], # present
                  [1, 9]] # absent

    state_prior: [1, 1]
```

### State Map File

The state map file is only required for the doublet model.
It specifies what happens when different combination of cell genotypes are observed in the doublet i.e. AA + AB = AB.
This file is passed in using `--state_map_file` flag.

```
# Doublets for SNV
snv:
  0: [[0, 0]] # List of genotype pairs that give state 0 i.e. AA

  1: [[0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]] # List of genotype pairs that give state 1 i.e. AB

  2: [[2, 2]] # List of genotype pairs that give state 2 i.e. BB

Doublets for presence/absence
breakpoint:
  0: [[0, 0]]

  1: [[0, 1], [1, 0], [1, 1]]
```

### Data files

Data files are stored as gzip compressed tsv files.
The data file should be a table.
The first column should be called cell_id and the remaining columns the names of the event ids i.e. SNV position.
Each row after the header should correspond to a cell with the observed genotype state filled in for each event using integer values.
For example for SNVs we use the encoding 0-AA, 1-AB and 2-BB.

### Sample file

The sample file is optional.
It is used to output estimates of sample specific cluster proportions.
The sample file is in tsv format and should have two columns `cell_id` and `sample`.
The cell_id should match that in the data file and the sample should be a unique sample name the cell came from.

### Labels file

The labels file is optional and used to provide an initial clustering of the data which will then be updated.
The sample file is in tsv format and should have two columns `cell_id` and `cluster`.

## Output files

For best results the scg model should be run multiple times with different random seeds using the `--seed` flag.
This is due to the fact that the optimization procedure only finds local optima, not global optima.
To avoid using a lot of disk space the scg model can be told only to save the lower bound for each restart using `--lower_bound_file`.
Once the best restart seed is found then the scg software can be rerun using that seed and full output can be generated using the `--out_dir` flag.
Note that output directory must exists.

The scg software will output the following files:

- cluster_posteriors.tsv.gz - The probability each cell is assigned to a cluster.
- genotype_posteriors.tsv.gz - The posterior probability of each genotype for a cluster.
- params.yaml - The inferred model parameters
- doublet_posteriors.tsv.gz - (doublet model only) The probability a cell is a doublet.
- double_cluster_posteriors.tsv.gz (doublet model only) The probability of which pair of clusters formed the doublet.
  There are K^2 possible pairs of double clusters where K is the number of clusters.
  The pairs are ordered (0, 0), (0, 1), (0, 2), ..., (K, K-1), (K, K).

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
