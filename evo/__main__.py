# -*- coding: utf-8 -*-
"""This module is responsible for running complete algorithms as tools, i.e. not
as a library in another program.
"""

import argparse
import re
import sys
import textwrap

import evo
import evo.gp
import evo.runners.bpgp as bpgp
import evo.sr
import evo.sr.backpropagation
import evo.sr.bpgp
import evo.utils
from evo.runners import text


class PreserveWhiteSpaceWrapRawTextHelpFormatter(
        argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog, indent_increment=2, max_help_position=24,
                 width=None):
        super().__init__(prog, indent_increment, max_help_position, width)

    @staticmethod
    def __add_whitespace(idx, i_w_space, text):
        if idx is 0:
            return text
        return (' ' * i_w_space) + text

    @staticmethod
    def __join_paragraphs(lines):
        newlines = []
        ni = -1
        first = True
        for i, line in enumerate(lines):
            if line.strip() is '':
                if first:
                    first = False
                else:
                    newlines.append('')
                ni = -1
            else:
                if ni == -1:
                    newlines.append(line)
                    ni = len(newlines) - 1
                else:
                    newlines[ni] += ' ' + line.strip()
                first = True
        return newlines

    def _split_lines(self, text, width):
        text_rows = text.splitlines()
        text_rows = self.__join_paragraphs(text_rows)
        for idx, line in enumerate(text_rows):
            search = re.search('\s*[0-9\-*]*\.?\s*', line)
            if line.strip() is '':
                text_rows[idx] = ' '
            elif search:
                l_w_space = search.end()
                lines = [self.__add_whitespace(i, l_w_space, x) for i, x in
                         enumerate(textwrap.wrap(line, width))]
                text_rows[idx] = lines

        return [item for sublist in text_rows for item in sublist]


class RootParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            prog='evo',
            description='Evolutionary computation package.',
            epilog='version: {}'.format(evo.__version__),
            formatter_class=PreserveWhiteSpaceWrapRawTextHelpFormatter
        )
        # common options
        parser.add_argument('--version',
                            help=text('Print version and exit.'),
                            action='version',
                            version=evo.__version__)
        parser.add_argument('--logconf',
                            help=text('logging configuration file (yaml '
                                      'format)'),
                            default=None)

        self.parser = parser

        # subcommands
        subparsers = parser.add_subparsers(title='algorithms',
                                           metavar='<algorithm>',
                                           dest='algorithm')
        self.parser_handlers = {p: h for p, h in [
            bpgp.create_parser(subparsers)
        ]}

    def parse(self):
        return self.parser.parse_args()

    def handle(self, ns: argparse.Namespace):
        return self.parser_handlers[ns.algorithm](ns)


def main():
    print('Arguments: {}'.format(sys.argv[1:]), file=sys.stderr)
    parser = RootParser()
    ns = parser.parse()
    status = parser.handle(ns)
    if status is not None:
        return status


if __name__ == '__main__':
    sys.exit(main())
