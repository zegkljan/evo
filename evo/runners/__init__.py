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
