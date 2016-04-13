# -*- coding: utf8 -*-
"""This module is responsible for running complete algorithms as tools, i.e. not
as a library in another program.
"""

import argparse
import textwrap
import random
import numpy
import re
import yaml
import logging.config
from pkg_resources import resource_stream

import evo
import evo.gp
import evo.sr
import evo.sr.backpropagation
import evo.sr.bpgp
import evo.utils


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


def parse():
    def text(t):
        return textwrap.dedent(t).strip()

    def prob(x):
        p = float(x)
        if p < 0 or p > 1:
            raise argparse.ArgumentTypeError('{} is not in range [0, 1]'
                                             .format(x))
        return p

    def bint(lb: int, ub: int=None):
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
                                                 .format(lbtext, ubtext))
            return n
        return f

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

    # task selection
    parser.add_argument('--task',
                        help=text('''
                            Specifies the kind of task to solve.
                            Possible values are:

                                * sr'''),
                        choices=['sr'],
                        required=True,
                        metavar='task')

    # algorithm selection
    parser.add_argument('-a', '--algorithm',
                        help=text('''
                            Algorithm to use. Possible values are:

                                bpgp.'''),
                        choices=['bpgp'],
                        required=True,
                        metavar='alg')

    # algorithm settings
    parser.add_argument('-s', '--seed',
                        help=text('Seed for random number generator.'),
                        type=int,
                        default=None,
                        metavar='n')
    parser.add_argument('-g', '--generations',
                        help=text('The number of generations to run for.'),
                        type=bint(1),
                        default=None,
                        required=True,
                        metavar='n')
    parser.add_argument('-p', '--pop-size',
                        help=text('Population size.'),
                        type=bint(1),
                        required=True,
                        metavar='n')
    parser.add_argument('-e', '--elitism',
                        help=text('Number of elites.'),
                        type=bint(0),
                        default=0,
                        metavar='n')
    parser.add_argument('-t', '--tournament-size',
                        help=text('Number of individuals competing in a '
                                  'tournament selection.'),
                        type=bint(2),
                        default=4,
                        metavar='n')
    parser.add_argument('--crossover-prob',
                        help=text(
                            'Probability of doing a crossover. Must be in '
                            'range [0, 1].'),
                        type=prob,
                        metavar='p')
    parser.add_argument('--mutation-prob',
                        help=text(
                            'Probability of doing a mutation. Must be in '
                            'range [0, 1].'),
                        type=prob,
                        metavar='p')
    parser.add_argument('--subtree-mutation-depth',
                        help=text('Maximum depth for newly generated subtrees '
                                  'in subtree mutation'),
                        type=bint(1),
                        default=4,
                        metavar='d')
    parser.add_argument('--rhh-min-depth',
                        help=text(
                            'Minimum depth of the Ramped Half\'n\'Half '
                            'initialisation procedure.'),
                        type=bint(1),
                        default=1,
                        metavar='d')
    parser.add_argument('--rhh-max-depth',
                        help=text(
                            'Maximum depth of the Ramped Half\'n\'Half '
                            'initialisation procedure.'),
                        type=bint(1),
                        default=6,
                        metavar='d')
    parser.add_argument('--bpgp-update-steps',
                        help=text('Number of iterations the weights tuning '
                                  'algorithm is allowed to do performe.'),
                        type=bint(0),
                        default=5,
                        metavar='n')
    parser.add_argument('--bpgp-updater',
                        help=text(
                            '''Updater for the backpropagation-GP. Possible
                            choices are:

                                * rprop+ - the Rprop+ (Rprop plus) algorithm

                                * rprop- - the Rprop- (Rprop minus) algorithm

                                * irprop+ - the iRprop+ (improved Rprop plus)
                                algorithm

                                * irprop- - the iRprop- (improved Rprop minus)
                                algorithm'''),
                        type=str,
                        choices=['rprop+', 'rprop-', 'irprop+', 'irprop-'],
                        default='irprop-')
    parser.add_argument('--rprop-delta-init',
                        help=text('Initial delta for every weight in the '
                                  'Rprop tuning algorithms (and its variants)'),
                        type=float,
                        default=0.1,
                        metavar='delta')
    parser.add_argument('--nodes',
                        help=text('''
                            Specifies the nodes that are available to the
                            algorithm to work with. Allowed values are:


                                * Add2 - binary addition

                                * Sub2 - binary subtraction

                                * Mul2 - binary multiplication

                                * Div2 - binary division

                                * Sin - sine, i.e. sin(x)

                                * Cos - cosine, i.e. cos(x)

                                * Exp - exponential function, i.e. e^x

                                * Abs - absolute value, i.e. |x|

                                * Power(N) (replace N by an integer) - N-th
                                power, i.e. x^N

                                * Sigmoid - sigmoid or logistic function,
                                i.e. 1 / (1 + e^-x)

                                * Sinc - the sinc function, i.e. sin(x) / x
                                and 1 for x = 0

                                * Softplus - the softplus function, i.e.
                                ln(1 + e^x)

                                * Const(X) (replace X by a number) - a
                                constant of the given value

                                * Const(A,B,C) (replace A, B and C by
                                numbers) - a constant that will be sampled
                                from a uniform distribution of a range [A,
                                B] C-times, i.e. there will be C
                                independently sampled constants available


                            Any value other than one of these is regarded as a
                            name of a variable (see option --x-mapping) and
                            results to a node for the corresponding
                            variable.'''),
                        type=str,
                        nargs='+',
                        required=True,
                        metavar='node')

    # input data setup
    parser.add_argument('-x', '--training-inputs',
                        help=text(
                            'Path to a CSV file containing the training '
                            'inputs, i.e. the input features.'),
                        type=str,
                        required=True,
                        metavar='filename')
    parser.add_argument('--x-mapping',
                        help=text('''
                            Mapping of the variables in the training inputs
                            file. There are two possibilities of the mapping
                            specification.


                            n1:x1,n2:x2,...  In this case, n1, n2, etc. are the
                            numbers of columns (zero-based) in the input file
                            (see option -x) and x1, x2, etc. are the names that
                            should be given to the corresponding input
                            variables in the resulting models (and their
                            names cannot contain a comma).


                            a:b or a: or :b or :  In this case, the columns
                            that will be used are consecutive columns from
                            number a (inclusive) to number b (exclusive).
                            Negative numbers can be used as with python list
                            indexing. Variables are then named x0, x1,
                            x2, ...

                            If some of the parts are missing then all the
                            columns in the respective direction will be used.
                            Examples: 1:3 are columns 1, 2; 0:-1 are all
                            columns except the last one; 3: are all columns
                            except the first three` :5 are first five
                            columns; : are all columns.


                            To distinguish between the two cases if only
                            one column is needed, the first case must end
                            with a comma, i.e. "0:x0," is equivalent to
                            "0:1"'''),
                        type=str,
                        default=None,
                        metavar='mapping')
    parser.add_argument('-y', '--training-outputs',
                        help=text(
                            'Path to a CSV file containing the training '
                            'outputs, i.e. the target values to fit. If the '
                            'file contains multiple columns then the last one'
                            'is assumed to be the one to use unless specified'
                            'otherwise via --y-mapping option.'),
                        type=str,
                        required=True,
                        metavar='filename')
    parser.add_argument('--y-mapping',
                        help=text(
                            'Mapping of the variables in the training '
                            'outputs file. It is treated as a number of '
                            'a column (zero-based) in the output file (see'
                            'option -y) where the output values are stored.'),
                        type=int,
                        default=None,
                        metavar='n')

    return parser.parse_args()


class BpgpArgHandler(object):
    # noinspection PyUnresolvedReferences
    def __init__(self, ns: argparse.Namespace):
        self.seed = ns.seed
        self.generations = ns.generations
        self.pop_size = ns.pop_size
        self.elitism = ns.elitism
        self.tournament_size = ns.tournament_size
        self.xover_prob = self._get_prob(ns.crossover_prob, 'crossover '
                                                            'probability')
        self.mut_prob = self._get_prob(ns.mutation_prob, 'mutation probability')
        self.mut_depth = ns.subtree_mutation_depth
        self.rhh_min_depth = ns.rhh_min_depth
        self.rhh_max_depth = ns.rhh_max_depth
        self.nodes_functions = self._get_nodes(ns.nodes)
        self.update_steps = ns.bpgp_update_steps
        self.updater = ns.bpgp_updater
        self.rprop_delta_init = ns.rprop_delta_init
        self.x, self.var_mapping = self._get_x(ns.training_inputs, ns.x_mapping)
        self.y = self._get_y(ns.training_outputs, ns.y_mapping)

    @staticmethod
    def _get_prob(prob, thing):
        if 0 <= prob <= 1:
            return prob
        raise ValueError('Probability ({}) must be in range [0, 1].'.format(
            thing))

    @staticmethod
    def _get_nodes(nodes: list):
        nodes_list = []
        for n in nodes:
            if n == 'Add2':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Add2(cache=True)))
                continue
            if n == 'Div2':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Div2(cache=True)))
                continue
            if n == 'Mul2':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Mul2(cache=True)))
                continue
            if n == 'Sub2':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Sub2(cache=True)))
                continue
            if n == 'Sin':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Sin(cache=True)))
                continue
            if n == 'Cos':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Cos(cache=True)))
                continue
            if n == 'Exp':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Exp(cache=True)))
                continue
            if n == 'Abs':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Abs(cache=True)))
                continue
            if n == 'Sigmoid':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Sigmoid(cache=True)))
                continue
            if n == 'Sinc':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Sinc(cache=True)))
                continue
            if n == 'Softplus':
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Softplus(cache=True)))
                continue
            match = re.fullmatch('Power\((.*)\)', n)
            if match:
                p = float(match.group(1))
                nodes_list.append((
                    None, lambda: evo.sr.backpropagation.Power(power=p,
                                                               cache=True)))
                continue
            match = re.fullmatch('Const\((.*),(.*),(.*)\)', n)
            if match:
                a = float(match.group(1))
                b = float(match.group(2))
                c = float(match.group(3))
                nodes_list.append((
                    'generator', lambda gen: [lambda: evo.sr.Const(
                        gen.uniform(a, b), cache=True) for _ in range(c)]))
                continue

            match = re.fullmatch('Const\((.*)\)', n)
            if match:
                c = float(match.group(1))
                nodes_list.append((None, lambda: evo.sr.Const(c, cache=True)))
                continue

            nodes_list.append((None, lambda: evo.sr.Variable(n)))
        return nodes_list

    @staticmethod
    def _get_x(filename, mapping_str):
        if ',' in mapping_str:
            substrs = mapping_str.split(',')
            file_col_mapping = {int(sub.split(':', 2)[0]): sub.split(':', 2)[1]
                                for sub in substrs}
            cols = sorted(file_col_mapping.keys())
            data = numpy.loadtxt(filename, delimiter=',', usecols=cols)
            var_mapping = {i: file_col_mapping[cols[i]]
                           for i in range(len(cols))}
        else:
            a, b = mapping_str.split(':', 2)
            if a == '':
                a = None
            else:
                a = int(a)
            if b == '':
                b = None
            else:
                b = int(b)
            data = numpy.loadtxt(filename, delimiter=',')
            data = data[:, slice(a, b)]
            var_mapping = {i: 'x{}'.format(i) for i in range(data.shape[1])}
        if data.ndim == 1:
            data = data[:, numpy.newaxis]
        return data, var_mapping

    @staticmethod
    def _get_y(filename, col_no):
        data = numpy.loadtxt(filename, delimiter=',')
        if data.ndim == 1:
            if col_no is not None and int(col_no) != 0:
                raise ValueError('Outputs file contains only a single column '
                                 'but a non-zero column specified in '
                                 '--y-mapping option.')
            return data
        else:
            if col_no is None:
                n = -1
            else:
                n = int(col_no)
            if n >= data.shape[1]:
                raise ValueError('Column number specified in --y-mapping '
                                 'option exceeds the number of columns in the '
                                 'outputs file.')
            return data[:, -1]


# noinspection PyUnresolvedReferences
def run(ns: argparse.Namespace):
    if ns.algorithm == 'bpgp':
        h = BpgpArgHandler(ns)
        run_sr_bpgp(h.seed, h.x, h.y, h.var_mapping, h.nodes_functions,
                    h.pop_size, h.elitism, h.tournament_size,
                    h.rhh_min_depth, h.rhh_max_depth, h.generations,
                    h.xover_prob, h.mut_prob, h.mut_depth, h.update_steps,
                    h.updater, h.rprop_delta_init)


def run_sr_bpgp(seed: int, x, y, var_mapping, node_creators, pop_size,
                elites_num, tournament_size, min_init_depth, max_init_depth,
                generations, xover_prob, mut_prob, subtree_mut_depth,
                update_steps, updater, delta_init):
    gen = random.Random(seed)
    if updater == 'rprop+':
        upd = evo.sr.backpropagation.RpropPlus(delta_init=delta_init)
    elif updater == 'rprop-':
        upd = evo.sr.backpropagation.RpropMinus(delta_init=delta_init)
    elif updater == 'irprop+':
        upd = evo.sr.backpropagation.IRpropPlus(delta_init=delta_init)
    elif updater == 'irprop-':
        upd = evo.sr.backpropagation.IRpropMinus(delta_init=delta_init)

    fit = evo.sr.bpgp.RegressionFitness(numpy.inf, [], x, y, var_mapping, upd,
                                        update_steps)
    funcs = []
    terms = []
    for info, f in node_creators:
        if info is None:
            n = f()
            if n.get_arity() == 0:
                terms.append(f)
            else:
                funcs.append(f)
        elif info == 'generator':
            ns = f(gen)
            for ff in ns:
                n = ff()
                if n.get_arity() == 0:
                    terms.append(ff)

    alg = evo.gp.Gp(
        fitness=fit,
        pop_strategy=evo.GenerationalPopulationStrategy(pop_size, elites_num),
        selection_strategy=evo.TournamentSelectionStrategy(tournament_size, gen,
                                                           fit),
        population_initializer=evo.gp.support.RampedHalfHalfInitializer(
            funcs,
            terms,
            min_depth=min_init_depth,
            max_depth=max_init_depth,
            generator=gen
        ),
        functions=funcs,
        terminals=terms,
        stop=generations,
        generator=gen,
        crossover_prob=xover_prob,
        mutation_prob=mut_prob,
        mutation_type=('subtree', subtree_mut_depth)
    )
    alg.run()
    print('Resulting model: {}'.format(fit.bsf.genotype.full_infix()))
    return fit.get_bsf()


def main():
    ns = parse()
    confdict = dict()
    with resource_stream('evo.resources', 'logging-default.yaml') as logconf:
        confdict = yaml.load(logconf)
    #evo.utils.nested_update(confdict['loggers'], {
    #    'evo.sr': {
    #        'level': 'DEBUG'
    #    }
    #})
    logging.config.dictConfig(confdict)
    run(ns)


if __name__ == '__main__':
    main()
