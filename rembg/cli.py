import click

from . import __version__
from .commands import command_functions


@click.group()
@click.version_option(version=__version__)
def _main() -> None:
    pass


for command in command_functions:
    _main.add_command(command)

_main()
