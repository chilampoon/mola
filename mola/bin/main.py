import click

# import subcommands
from .read import mola_read
from .mutation import mola_mut
from .infer import mola_infer
from .parse import mola_parse

HELP_CONTEXT = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=HELP_CONTEXT)
def mola():
    """
    🌞🐟
    """
    pass

mola.add_command(mola_read)
mola.add_command(mola_mut)
mola.add_command(mola_infer)
mola.add_command(mola_parse)

if __name__ == '__main__':
    mola()
