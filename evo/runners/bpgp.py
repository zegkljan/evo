import argparse
import logging
import logging.config
import os
import sys
import time

import numpy as np
import yaml
from pkg_resources import resource_stream

import evo.utils
from evo.runners import text, bounded_integer, bounded_float, float01


def create_bpgp_parser(subparsers):
    parser = subparsers.add_parser('bpgp',
                                   help='BackPropagation Genetic Programming')

    # input data setup
    setup_input_data_arguments(parser)

    # output data setup
    setup_output_data_arguments(parser)

    # general settings
    setup_general_settings_arguments(parser)

    # algorithm parameters
    setup_parameters_arguments(parser)


def setup_input_data_arguments(parser):
    parser.add_argument('-d', '--training-data',
                        help=text('Path to a CSV file containing the training '
                                  'data, i.e. the input features and output '
                                  'values. By default, all columns except the '
                                  'last one are considered to be the features '
                                  'and the last column is considered to be the '
                                  'target value. Can be overriden by options '
                                  '--x-columns, --y-data and --y-column.'),
                        type=str,
                        required=True,
                        metavar='filename')
    parser.add_argument('--x-columns',
                        help=text('Specifies which columns in the training '
                                  'data file are to be used as features. It '
                                  'is a space separated list of numbers. Each '
                                  'number can be prefixed with the character '
                                  '"^" to denote that this column is NOT a '
                                  'feature column. If there are only regular '
                                  'numbers (i.e. without ^), then these are '
                                  'the numbers used as the features. If there '
                                  'are only exclusion numbers (i.e. with ^) '
                                  'then all the columns except those specified '
                                  'are used as the features. If there are both '
                                  'regular and exclusion numbers then the '
                                  'features are all the columns specified by '
                                  'the regular numbers except those specified '
                                  'by the exclusion numbers (i.e. exclusion '
                                  'numbers that are not among regular numbers '
                                  'have no effect).'),
                        type=str,
                        nargs='*')
    parser.add_argument('-y', '--y-data',
                        help=text(
                            'Path to a CSV file containing the training '
                            'outputs, i.e. the target values to fit. If the '
                            'file contains multiple columns then the last one'
                            'is assumed to be the one to use. This can be '
                            'overridden by the argument --y-column.'),
                        type=str,
                        metavar='filename')
    parser.add_argument('--y-column',
                        help=text('Determines which column from a data file is '
                                  'to be used as the target values. If '
                                  '--y-data is specified then it is from that '
                                  'file, otherwise it is from the file '
                                  'specified by --training-data.'),
                        type=int,
                        default=None,
                        metavar='n')
    parser.add_argument('--delimiter',
                        help=text('Field delimiter of the CSV files specified '
                                  'in --training-data and --y-data. Default '
                                  'is ",".'),
                        type=str,
                        default=',')


def setup_output_data_arguments(parser):
    parser.add_argument('-o', '--output-directory',
                        help=text('Directory to which the output files will be '
                                  'written. Default is current directory.'),
                        default='.')


def setup_general_settings_arguments(parser):
    parser.add_argument('--seed',
                        help=text('Seed for random number generator. If not '
                                  'specified, current time will be used.'),
                        type=int,
                        default=int(10 * time.time()),
                        metavar='n')
    parser.add_argument('--generations',
                        help=text('The maximum number of generations to run '
                                  'for.'),
                        type=bounded_integer(1),
                        default=float('inf'))
    parser.add_argument('--time',
                        help=text('The maximum number of seconds to run for.'),
                        type=bounded_float(0),
                        default=float('inf'))
    parser.add_argument('--generation-time-combinator',
                        help=text('If both --generations and --time are '
                                  'specified, this determines how are the two '
                                  'conditions combined. The value of "any" '
                                  'causes termination when any of the two '
                                  'conditions is met. The value of "all" '
                                  'causes termination only after both '
                                  'conditions are met. Default is "any".'),
                        choices=['any', 'all'],
                        default='any')


def setup_parameters_arguments(parser):
    parser.add_argument('--pop-size',
                        help=text('population size'),
                        type=bounded_integer(1),
                        default=100)
    parser.add_argument('--elitism',
                        help=text('Number of elites as a fraction (float '
                                  'between 0 and 1) of the population size. '
                                  'Default is 0.15.'),
                        type=float01(),
                        default=0.15)
    parser.add_argument('--tournament-size',
                        help=text('Number of individuals competing in a '
                                  'tournament selection as a fraction (float '
                                  'between 0 and 1) of the population size. '
                                  'Default is 0.1.'),
                        type=float01(),
                        default=0.1)
    parser.add_argument('--max-genes',
                        help=text('Maximum number of genes. Default is 4.'),
                        type=bounded_integer(1),
                        default=4)
    parser.add_argument('--max-depth',
                        help=text('Maximum depth of a gene. Default is 5.'),
                        type=bounded_integer(1),
                        default=5)
    parser.add_argument('--max-nodes',
                        help=text('Maximum number of nodes in a gene. Default '
                                  'is infinity (i.e. unbounded).'),
                        type=bounded_integer(1),
                        default=float('inf'))
    parser.add_argument('--crossover-prob',
                        help=text('Probability of crossover. Default is 0.84'),
                        type=float01(),
                        default=0.84)
    parser.add_argument('--highlevel-crossover-prob',
                        help=text('Probability of choosing a high-level '
                                  'crossover as a crossover operation. The '
                                  'complement to 1 is then the probability of '
                                  'subtree crossover. Default is 0.2.'),
                        type=float01(),
                        default=0.2)
    parser.add_argument('--highlevel-crossover-rate',
                        help=text('Probability that a gene is chosen for '
                                  'crossover in high-level crossover. Default '
                                  'is 0.5.'),
                        type=float01(),
                        default=0.5)
    parser.add_argument('--mutation-prob',
                        help=text('Probability of mutation. Default is 0.14.'),
                        type=float01(),
                        default=0.14)
    parser.add_argument('--constant-mutation-prob',
                        help=text('Probability of choosing mutation of '
                                  'constants as a mutation operation. The '
                                  'complement to 1 of this parameter and of '
                                  '--weights-muatation-prob is then the '
                                  'probability of subtree mutation. To turn '
                                  'this mutation off, set the parameter to 0. '
                                  'Default is 0.05.'),
                        type=float01(),
                        default=0.05)
    parser.add_argument('--constant-mutation-sigma',
                        help=text('Standard deviation of the normal '
                                  'distribution used to mutate the constant '
                                  'values. Default is 0.1.'),
                        type=bounded_float(0),
                        default=0.1)
    parser.add_argument('--weights-mutation-prob',
                        help=text('Probability of choosing mutation of '
                                  'weights as a mutation operation. The '
                                  'complement to 1 of this parameter and of '
                                  '--constant-muatation-prob is then the '
                                  'probability of subtree mutation. To turn '
                                  'this mutation off, set the parameter to 0. '
                                  'Default is 0.05.'),
                        type=float01(),
                        default=0.05)
    parser.add_argument('--weights-mutation-sigma',
                        help=text('Standard deviation of the normal '
                                  'distribution used to mutate the weights. '
                                  'Default is 3.'),
                        type=bounded_float(0),
                        default=3)
    parser.add_argument('--backpropagation-mode',
                        help=text('How is backpropagation used. '
                                  'Mode "none" turns off the backpropagation '
                                  'completely. Mode "raw" means that the '
                                  'number of steps is always the number '
                                  'specified in --backpropagation-steps (and '
                                  'hence --min-backpropagation-steps is '
                                  'ignored). Modes "nodes" and "depth" mean '
                                  'that the number of steps is the number '
                                  'specified in --backpropagation-steps minus '
                                  'the total number of nodes of the individual '
                                  '(for "nodes") of the maximum depth of the '
                                  'genes (for "depth"). Default is "none", '
                                  'i.e. no backpropagation.'),
                        choices=['none', 'raw', 'nodes', 'depth'],
                        default='none')
    parser.add_argument('--backpropagation-steps',
                        help=text('How many backpropagation steps are '
                                  'performed per evaluation. The actual number '
                                  'is computed based on the value of '
                                  '--backpropagation-mode. Default is 25.'),
                        type=bounded_integer(0),
                        default=25)
    parser.add_argument('--min-backpropagation-steps',
                        help=text('At least this number of backpropagation is'
                                  'always performed, no matter what '
                                  '--backpropagation-steps and '
                                  '--backpropagation-mode are set to (except '
                                  'for "none" mode). Default is 2'),
                        type=bounded_integer(0),
                        default=2)
    parser.add_argument('--weighted',
                        help=text('If specified, the inner nodes will be '
                                  'weighted, i.e. with multiplicative and '
                                  'additive weights, tunable by '
                                  'backpropagation and weights mutation.'),
                        action='store_true')
    parser.add_argument('--lcf-mode',
                        help=text('How are the LCFs used. '
                                  'Mode "none" turns off the LCFs completely. '
                                  'Mode "unsynced" means that each LCF is free '
                                  'to change on its own (by backpropagation '
                                  'and/or mutation). Mode "synced" means that '
                                  'the LCFs are synchronized across the '
                                  'individual. Mode "global" means that the '
                                  'LCFs are synchronized across the whole '
                                  'population. Default is "none", i.e. no '
                                  'LCFs.'),
                        choices=['none', 'unsynced', 'synced', 'global'],
                        default='none')
    parser.add_argument('--weight-init',
                        help=text('How are weights in weighted nodes and LCFs '
                                  '(if they are turned on) initialized. Mode '
                                  '"latent" means that the initial values of '
                                  'weights are such that they play no role, '
                                  'i.e. additive weights set to zero, '
                                  'multiplicative weights set to one (or only '
                                  'one of them in case of LCFs). Mode "random" '
                                  'means that the values of weights are chosen '
                                  'randomly (see option --random-init-bounds). '
                                  'Default is "latent".'),
                        choices=['latent', 'random'],
                        default='latent')
    parser.add_argument('--random-init-bounds',
                        help=text('Bounds of the range the weights are sampled '
                                  'when --weight-init is set to "random".'),
                        nargs=2,
                        type=float)


# noinspection PyUnresolvedReferences
def handle(ns: argparse.Namespace):
    with resource_stream('evo.resources', 'logging-default.yaml') as f:
        logging_conf = yaml.load(f)
    if ns.logconf is not None:
        if not os.path.isfile(ns.logconf):
            print('Supplied logging configuration file does not exist or is '
                  'not a file. Exitting.', file=sys.stderr)
            sys.exit(1)
        with open(ns.logconf) as f:
            local = yaml.load(f)
        evo.utils.nested_update(logging_conf, local)
    logging.config.dictConfig(logging_conf)
    logging.info('Starting evo.')
    logging.info('Loading training data...')
    x_data, y_data = load_data(ns.training_data, ns.y_data, ns.x_columns,
                               ns.y_column, ns.delimiter)
    logging.info('Training data loaded.')
    output = prepare_output(ns)


# noinspection PyUnresolvedReferences
def load_data(x_fn, y_fn, x_columns, y_column, delimiter):
    if not os.path.isfile(x_fn):
        print('File {} does not exist or is not a file. Exitting.'.format(x_fn),
              file=sys.stderr)
        sys.exit(1)
    logging.info('Data file: %s', x_fn)
    data = np.loadtxt(x_fn, delimiter=delimiter)
    if x_columns:
        included = [int(x) for x in x_columns if not x.startswith('^')]
        excluded = set(int(x[1:]) for x in x_columns if x.startswith('^'))
        included.sort()
        if not included:
            included = list(range(data.shape[1]))
        included = list(filter(lambda x: x not in excluded, included))
        logging.info('Data x columns: %s', included)
        x_data = data[:, included]
    else:
        x_data = data[:, :-1]

    if y_fn is not None:
        if not os.path.isfile(y_fn):
            print('File {} does not exist or is not a file. Exitting.'
                  .format(y_fn), file=sys.stderr)
            sys.exit(1)
        logging.info('Y-data file: %s', y_fn)
        data = np.loadtxt(y_fn, delimiter=delimiter)
    if y_column is not None:
        logging.info('Data y column: %i', y_column)
        y_data = data[:, y_column]
    else:
        y_data = data[:, -1]

    logging.info('X data shape: %s', x_data.shape)
    logging.info('Y data shape: %s', y_data.shape)

    return x_data, y_data


# noinspection PyUnresolvedReferences
def prepare_output(ns: argparse.Namespace):
    if os.path.isdir(ns.output_directory):
        logging.warning('Output directory %s already exists! Contents might be '
                        'overwritten', ns.output_directory)
    os.makedirs(ns.output_directory, exist_ok=True)
    logging.info('Output directory (relative): %s',
                 os.path.relpath(ns.output_directory, os.getcwd()))
    logging.info('Output directory (absolute): %s',
                 os.path.abspath(ns.output_directory))

    return {
        'y_trn': 'y_trn.txt',
        'summary': 'summary.txt',
        'm_func_templ': '{}_func.m'
    }
