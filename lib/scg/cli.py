import click

import scg.run


@click.command(
    context_settings={"max_content_width": 120},
    name="fit"
)
@click.option(
    "-i", "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to input file specifying the configuration in YAML format configuration file."""
)
@click.option(
    "-o", "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where output file be written in HDF5 format."""
)
@click.option(
    "-c", "--convergence-threshold",
    default=1e-4,
    type=float,
    help="""Maximum relative ELBO difference between iterations to decide on convergence."""
    """Default is 10^-4."""
)
@click.option(
    "-m", "--max-iters",
    default=int(1e4),
    type=int,
    help="""Maximum number of ELBO optimization iterations."""
    """Default is 10,0000."""
)
@click.option(
    "-s", "--seed",
    default=None,
    type=int,
    help="""Set random seed so results can be reproduced. By default a random seed is chosen."""
)
def fit(**kwargs):
    scg.run.fit(**kwargs)


@click.group(name="scg")
def main():
    pass


main.add_command(fit)
