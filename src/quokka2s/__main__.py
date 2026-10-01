"""Two entry points for numerical emission processing and local plotting."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog='quokka2s')
    parser.add_argument('command', choices=('process', 'plot'),
                        help='process a snapshot or plot saved products')
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.parse_args(arguments)
        return
    command = arguments.pop(0)
    if command not in ('process', 'plot'):
        parser.error(f'invalid command: {command}')
    if command == 'process':
        from .emission_processing import main as process
        process(arguments)
    else:
        from .emission_plots import main as plot
        plot(arguments)


if __name__ == '__main__':
    main()
