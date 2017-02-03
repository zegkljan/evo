import argparse
import textwrap


def text(t):
    return textwrap.dedent(t).strip()


def bounded_integer(lb: int, ub: int=None):
    if lb is None:
        lbtext = '-inf'
    else:
        lbtext = str(lb)
    if ub is None:
        ubtext = 'inf'
    else:
        ubtext = str(ub)

    def f(x):
        n = int(x)
        if (lb is not None and n < lb) or (ub is not None and n > ub):
            raise argparse.ArgumentTypeError('{} is not in range [{}, {}]'
                                             .format(x, lbtext, ubtext))
        return n

    return f


def bounded_float(lb: float, ub: float=float('inf')):
    def f(x):
        n = float(x)
        if n < lb or n > ub:
            raise argparse.ArgumentTypeError('{} is not in range [{}, {}]'
                                             .format(x, lb, ub))
        return n
    return f


def float01():
    return bounded_float(0, 1)


class DataSpec(object):
    def __init__(self, spec: str):
        self.file = None
        self.x_cols = None
        self.y_col = None

        parts = spec.split(':')
        if len(parts) == 1:
            self.file = parts[0]
        elif len(parts) == 3:
            self.file = parts[0]
            try:
                self.x_cols = list(map(int, parts[1].split(',')))
            except ValueError:
                raise argparse.ArgumentTypeError('x-columns spec is invalid')
            try:
                self.y_col = int(parts[2])
            except ValueError:
                raise argparse.ArgumentTypeError('y-column spec is invalid')
        else:
            raise argparse.ArgumentTypeError('Data specification contains '
                                             'invalid number of parts '
                                             '(separated by colon). The number '
                                             'of parts must be either 1 (no '
                                             'colon, only file name) or 3 (two '
                                             'colons, file name, x-columns '
                                             'spec and y-column spec).')

    def __repr__(self):
        if self.x_cols is None and self.y_col is None:
            return 'DataSpec(\'{}\')'.format(self.file)
        return 'DataSpec(\'{}:{}:{}\')'.format(self.file, self.x_cols,
                                               self.y_col)
