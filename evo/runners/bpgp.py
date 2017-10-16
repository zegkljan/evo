# -*- coding: utf-8 -*-
import argparse
import collections
import gc
import logging
import logging.config
import math
import os
import random
import re
import time

import numpy as np
import yaml
from pkg_resources import resource_stream

import evo.gp
import evo.sr.backpropagation
import evo.sr.bpgp
import evo.utils
from evo.runners import text, bounded_integer, bounded_float, float01, \
    DataSpec, PropagateExit


class Runner(object):
    PARSER_ARG = 'bpgp'

    def __init__(self, subparsers):
        self.parser = subparsers.add_parser(
            self.PARSER_ARG, help='BackPropagation Genetic Programming',
            conflict_handler='resolve')

        # input data setup
        self.setup_input_data_arguments()

        # output data setup
        self.setup_output_data_arguments()

        # general settings
        self.setup_general_settings_arguments()

        # algorithm parameters
        self.setup_parameters_arguments()

    def setup_input_data_arguments(self):
        self.parser.add_argument(
            '--training-data',
            help=text(
                'Specification of the training data in the '
                'format file[:x-columns:y-column].\n\n'
                '"file" is a path to a CSV file to be '
                'loaded.\n\n'
                '"x-columns" is a comma-separated (only comma, '
                'without whitespaces) list of numbers of '
                'columns (zero-based) to be used as the '
                'features.\n\n'
                '"y-column" is a number of column that will be '
                'used as the target. You can use negative '
                'numbers to count from back, i.e. -1 is the '
                'last column, -2 is the second to the last '
                'column, etc.\n\n'
                'The bracketed part is optional and can be '
                'left out. If it is left out, all columns '
                'except the last one are used as features and '
                'the last one is used as target.'),
            type=DataSpec,
            required=True,
            metavar='file[:x-columns:y-column]')
        self.parser.add_argument(
            '--testing-data',
            help=text(
                'Specification of the testing data. The format '
                'is identical to the one of --training-data. '
                'The testing data must have the same number of '
                'columns as --training-data.\n\n'
                'The testing data are evaluated with the best '
                'individual after the evolution finishes. If, '
                'for some reason, the individual fails to '
                'evaluate (e.g. due to numerical errors like '
                'division by zero), the second best individual '
                'is tried and so forth until the individual '
                'evaluates or there is no individual left.\n\n'
                'If the testing data is not specified, no '
                'testing is done (only training measures are '
                'reported).'),
            type=DataSpec,
            required=False,
            metavar='file[:x-columns:y-column]')
        self.parser.add_argument(
            '--delimiter',
            help=text(
                'Field delimiter of the CSV files specified in '
                '--training-data and --testing-data. Default '
                'is ",".'),
            type=str,
            default=',')

    def setup_output_data_arguments(self):
        self.parser.add_argument(
            '-o', '--output-directory',
            help=text(
                'Directory to which the output files will be '
                'written. If "-" is specified, no output will '
                'be written. If you need to output to a '
                'directory literally named "-", specify an '
                'absolute or relative path (e.g. "./-"). '
                'Default is the current directory.'),
            default='.')
        self.parser.add_argument(
            '--m-fun',
            help=text(
                'Name of the matlab function the model will be '
                'written to (without extension). Default is '
                '"func".'),
            type=str,
            default='func')
        self.parser.add_argument(
            '--output-string-template',
            help=text(
                'Template for the string that will be printed '
                'to the standard output at the very end of the '
                'algorithm. This can be used to report '
                'algorithm performance to tuners such as SMAC. '
                'Default is no string (nothing is printed).\n\n'
                'The string can contain any of the following '
                'placeholders: {tst_r2}, {trn_r2}, '
                '{tst_r2_inv}, {trn_r2_inv}, {tst_mse}, '
                '{trn_mse}, {tst_mae}, {trn_mae}, {tst_wae}, '
                '{trn_wae}, {runtime}, {seed}, {iterations}.'),
            type=str,
            default=None)

    def setup_general_settings_arguments(self):
        self.parser.add_argument(
            '--seed',
            help=text(
                'Seed for random number generator. If not '
                'specified, current time will be used.'),
            type=int,
            default=int(10 * time.time()),
            metavar='n')
        self.parser.add_argument(
            '--generations',
            help=text(
                'The maximum number of generations to run for. '
                'Default is infinity (i.e. until stopped '
                'externally or with some other stopping '
                'condition).'),
            type=bounded_integer(1),
            default=float('inf'))
        self.parser.add_argument(
            '--time',
            help=text(
                'The maximum number of seconds to run for. '
                'Default is infinity (i.e. until stopped '
                'externally or with some other stopping '
                'condition).'),
            type=bounded_float(0),
            default=float('inf'))
        self.parser.add_argument(
            '--generation-time-combinator',
            help=text(
                'If both --generations and --time are '
                'specified, this determines how are the two '
                'conditions combined. The value of "any" '
                'causes termination when any of the two '
                'conditions is met. The value of "all" causes '
                'termination only after both conditions are '
                'met. Default is "any".'),
            choices=['any', 'all'],
            default='any')

    def setup_parameters_arguments(self):
        self.parser.add_argument(
            '--pop-size',
            help=text('Population size. Default is 100.'),
            type=bounded_integer(1),
            default=100)
        self.parser.add_argument(
            '--elitism',
            help=text(
                'Number of elites as a fraction (float between '
                '0 and 1) of the population size. Default is '
                '0.15.'),
            type=float01(),
            default=0.15)
        self.parser.add_argument(
            '--tournament-size',
            help=text(
                'Number of individuals competing in a '
                'tournament selection as a fraction (float '
                'between 0 and 1) of the population size. '
                'Default is 0.1.'),
            type=float01(),
            default=0.1)
        self.parser.add_argument(
            '--max-genes',
            help=text('Maximum number of genes. Default is 4.'),
            type=bounded_integer(1),
            default=4)
        self.parser.add_argument(
            '--max-depth',
            help=text('Maximum depth of a gene. Default is 5.'),
            type=bounded_integer(1),
            default=5)
        self.parser.add_argument(
            '--max-nodes',
            help=text(
                'Maximum number of nodes in a gene. Default is '
                'infinity (i.e. unbounded).'),
            type=bounded_integer(1),
            default=float('inf'))
        self.parser.add_argument(
            '--functions',
            help=text(
                'A comma-separated (without whitespaces) list '
                'of functions available to the algorithm. '
                'Available functions are: Add2, Sub2, Mul2, '
                'Div2, Sin, Cos, Exp, Abs, Sqrt, Sigmoid, '
                'Tanh, Sinc, Softplus, Gauss, BentIdentity, '
                'Pow(n) where n is the positive integer power. '
                'Default is Add2,Sub2,Mul2,Sin,Cos,Exp,Sigmoid,'
                'Tanh,Sinc,Softplus,Gauss,Pow(2),Pow(3),Pow(4),'
                'Pow(5),Pow(6)'),
            type=str,
            metavar='Function[,Function ...]',
            default='Add2,Sub2,Mul2,Sin,Cos,Exp,Sigmoid,Tanh,'
                    'Sinc,Softplus,Gauss,Pow(2),Pow(3),Pow(4),'
                    'Pow(5),Pow(6)')
        self.parser.add_argument(
            '--crossover-prob',
            help=text(
                'Probability of crossover. Default is 0.84'),
            type=float01(),
            default=0.84)
        self.parser.add_argument(
            '--highlevel-crossover-prob',
            help=text(
                'Probability of choosing a high-level '
                'crossover as a crossover operation. The '
                'complement to 1 is then the probability of '
                'subtree crossover. If --max-genes is 1, this '
                'parameter is ignored (even if not specified) '
                'and set to 0. Default is 0.2.'),
            type=float01(),
            default=0.2)
        self.parser.add_argument(
            '--highlevel-crossover-rate',
            help=text(
                'Probability that a gene is chosen for '
                'crossover in high-level crossover. Default is '
                '0.5.'),
            type=float01(),
            default=0.5)
        self.parser.add_argument(
            '--mutation-prob',
            help=text(
                'Probability of mutation. Default is 0.14.'),
            type=float01(),
            default=0.14)
        self.parser.add_argument(
            '--constant-mutation-prob',
            help=text(
                'Probability of choosing mutation of constants '
                'as a mutation operation. The complement to 1 '
                'of this parameter and of '
                '--weights-muatation-prob is then the '
                'probability of subtree mutation. To turn this '
                'mutation off, set the parameter to 0. Default '
                'is 0.05.'),
            type=float01(),
            default=0.05)
        self.parser.add_argument(
            '--constant-mutation-sigma',
            help=text(
                'Standard deviation of the normal distribution '
                'used to mutate the constant values. Default '
                'is 0.1.'),
            type=bounded_float(0),
            default=0.1)
        self.parser.add_argument(
            '--weights-mutation-prob',
            help=text(
                'Probability of choosing mutation of weights '
                'as a mutation operation. The complement to 1 '
                'of this parameter and of '
                '--constant-muatation-prob is then the '
                'probability of subtree mutation. To turn this '
                'mutation off, set the parameter to 0. Default '
                'is 0.05.'),
            type=float01(),
            default=0.05)
        self.parser.add_argument(
            '--weights-mutation-sigma',
            help=text(
                'Standard deviation of the normal distribution '
                'used to mutate the weights. Default is 3.'),
            type=bounded_float(0),
            default=3)
        self.parser.add_argument(
            '--backpropagation-mode',
            help=text(
                'How backpropagation is used. Mode "none" '
                'turns the backpropagation off completely. '
                'Mode "raw" means that the number of steps is '
                'always the number specified in '
                '--backpropagation-steps (and hence '
                '--min-backpropagation-steps is ignored). '
                'Modes "nodes" and "depth" mean that the '
                'number of steps is the number specified in '
                '--backpropagation-steps minus the total '
                'number of nodes of the individual (for '
                '"nodes") or the maximum depth of the genes '
                '(for "depth"). Default is "none", i.e. no '
                'backpropagation.'),
            choices=['none', 'raw', 'nodes', 'depth'],
            default='none')
        self.parser.add_argument(
            '--backpropagation-steps',
            help=text(
                'How many backpropagation steps are performed '
                'per evaluation. The actual number is computed '
                'based on the value of --backpropagation-mode. '
                'Default is 25.'),
            type=bounded_integer(0),
            default=25)
        self.parser.add_argument(
            '--min-backpropagation-steps',
            help=text(
                'At least this number of backpropagation steps '
                'is always performed, no matter what '
                '--backpropagation-steps and '
                '--backpropagation-mode are set to (except for '
                '"none" mode). Default is 2.'),
            type=bounded_integer(0),
            default=2)
        self.parser.add_argument(
            '--weighted',
            help=text(
                'If specified, the inner nodes will be '
                'weighted, i.e. with multiplicative and '
                'additive weights, tunable by backpropagation '
                'and weights mutation.'),
            action='store_true')
        self.parser.add_argument(
            '--lcf-mode',
            help=text(
                'How the LCFs are used. Mode "none" turns the '
                'LCFs off completely. Mode "unsynced" means '
                'that each LCF is free to change on its own '
                '(by backpropagation and/or mutation). Mode '
                '"synced" means that the LCFs are synchronized '
                'across the individual. Mode "global" means '
                'that the LCFs are synchronized across the '
                'whole population. Default is "none", i.e. no '
                'LCFs.'),
            choices=['none', 'unsynced', 'synced', 'global'],
            default='none')
        self.parser.add_argument(
            '--weight-init',
            help=text(
                'How are weights in weighted nodes and LCFs '
                '(if they are turned on) initialized. Mode '
                '"latent" means that the initial values of '
                'weights are such that they play no role, i.e. '
                'additive weights set to zero, multiplicative '
                'weights set to one (or only one of them in '
                'case of LCFs). Mode "random" means that the '
                'values of weights are chosen randomly (see '
                'option --random-init-bounds). Default is '
                '"latent".'),
            choices=['latent', 'random'],
            default='latent')
        self.parser.add_argument(
            '--weight-init-bounds',
            help=text(
                'Bounds of the range the weights are sampled '
                'from when --weight-init is set to "random". '
                'Default is -10 and 10.'),
            nargs=2,
            type=float,
            metavar='bound',
            default=[-10, 10])
        self.parser.add_argument(
            '--const-init-bounds',
            help=text(
                'Bounds of the range the constants (leaf '
                'nodes) are sampled from. Default is -10 and '
                '10.'),
            nargs=2,
            type=float,
            metavar='bound',
            default=[-10, 10])
        self.parser.add_argument(
            '--constants',
            help=text(
                'Type constant leaf nodes work. One of the '
                'following: none, classical, tunable. The '
                '"none" type means no such nodes will be '
                'available. Type "classical" means that the '
                'constants will be generated randomly at '
                'initialization (and subtree mutation) and can '
                'be modified via mutation. The type "tunable" '
                'means the values of the constants are subject '
                'to gradient-based tuning.'),
            choices=['none', 'classical', 'tunable'],
            default='classical')

    def handle(self, ns: argparse.Namespace):
        try:
            return self.handle_wrapper(ns)
        except PropagateExit as e:
            return e.status

    def handle_wrapper(self, ns: argparse.Namespace):
        params = self.get_params(ns)

        with resource_stream('evo.resources', 'logging-default.yaml') as f:
            logging_conf = yaml.load(f)
        if params['logconf'] is not None:
            if not os.path.isfile(params['logconf']):
                logging.config.dictConfig(logging_conf)
                logging.error('Supplied logging configuration file does not '
                              'exist or is not a file. Exitting.')
                raise PropagateExit(1)
            with open(params['logconf']) as f:
                local = yaml.load(f)
            evo.utils.nested_update(logging_conf, local)
        logging.config.dictConfig(logging_conf)
        logging.info('Starting evo.')

        x_data_trn, y_data_trn, x_data_tst, y_data_tst = self.load_data(params)
        if x_data_tst is not None and x_data_trn.shape[1] != x_data_tst.shape[1]:
            logging.error('Training and testing data have different number of '
                          'columns. Exitting.')
            raise PropagateExit(1)
        output = self.prepare_output(params)

        self.log_params(params)
        # prepare stuff needed for algorithm creation
        rng = self.create_rng(params)
        functions = self.create_functions(rng, params)
        global_lcs, terminals = self.create_terminals(rng, x_data_trn, True,
                                                      params)
        fitness = self.create_fitness(params, x_data_trn, y_data_trn)
        crossover = self.create_crossover(rng, params)
        mutation = self.create_mutation(rng, functions, terminals, params)
        population_strategy = self.create_population_strategy(params)
        reproduction_strategy = self.create_reproduction_strategy(
            rng, crossover, mutation, functions, terminals, params)
        callback = self.create_callback(params)
        stopping_condition = self.create_stopping_condition(params)
        population_initializer = self.create_population_initializer(rng,
                                                                    functions,
                                                                    terminals,
                                                                    params)
        # create algorithm
        algorithm = self.create_algorithm(rng, functions, terminals, global_lcs,
                                          fitness, population_strategy,
                                          reproduction_strategy, callback,
                                          stopping_condition,
                                          population_initializer, params)

        # set numpy to raise an error for everything except underflow
        np.seterr(all='raise', under='warn')
        result = algorithm.run()
        self.postprocess(algorithm, x_data_trn, y_data_trn, x_data_tst,
                         y_data_tst, output, ns)
        if result:
            return 0
        return 100

    def load_data(self, params: dict):
        def load(ds: DataSpec, delimiter: str, prefix: str):
            if not os.path.isfile(ds.file):
                logging.error('%s data file %s does not exist or is not a '
                              'file. Exitting.', prefix, ds.file)
                raise PropagateExit(1)
            logging.info('%s data file: %s', prefix, ds.file)
            data = np.loadtxt(ds.file, delimiter=delimiter)
            if ds.x_cols is not None:
                logging.info('%s data x columns: %s', prefix, ds.x_cols)
                x_data = data[:, ds.x_cols]
            else:
                x_data = data[:, :-1]

            if ds.y_col is not None:
                logging.info('%s data y column: %i', prefix, ds.y_col)
                y_data = data[:, ds.y_col]
            else:
                y_data = data[:, -1]

            return x_data, y_data

        dlm = params['delimiter']
        training_ds = params['training-data']
        testing_ds = params['testing-data']

        training_x, training_y = load(training_ds, dlm, 'Training')
        logging.info('Training X data shape (rows, cols): %s', training_x.shape)
        logging.info('Training Y data shape (elements,): %s', training_y.shape)
        testing_x, testing_y = None, None
        if testing_ds is not None:
            testing_x, testing_y = load(testing_ds, dlm, 'Testing')
            logging.info('Testing X data shape (rows, cols): %s',
                         training_x.shape)
            logging.info('Testing Y data shape (elements,): %s',
                         training_y.shape)

        return training_x, training_y, testing_x, testing_y

    def prepare_output(self, params: dict):
        output_data = collections.defaultdict(lambda: None)

        if params['output_string_template'] is not None:
            output_data['output_string_template'] = \
                params['output_string_template']
        output_data['m_fun'] = params['m_fun']

        if params['output_directory'] is None:
            return output_data

        if params['output_directory'] == '-':
            logging.info('No output directory is used.')
            return output_data

        if os.path.isdir(params['output_directory']):
            logging.warning('Output directory %s already exists! Contents '
                            'might be overwritten', params['output_directory'])
        os.makedirs(params['output_directory'], exist_ok=True)
        logging.info('Output directory (relative): %s',
                     os.path.relpath(params['output_directory'], os.getcwd()))
        logging.info('Output directory (absolute): %s',
                     os.path.abspath(params['output_directory']))

        output_data['y_trn'] = os.path.join(params['output_directory'],
                                            'y_trn.txt')
        output_data['y_tst'] = os.path.join(params['output_directory'],
                                            'y_tst.txt')
        output_data['summary'] = os.path.join(params['output_directory'],
                                              'summary.txt')
        output_data['m_func_templ'] = os.path.join(params['output_directory'],
                                                   '{}.m')
        output_data['stats'] = os.path.join(params['output_directory'],
                                            'stats.csv')
        return output_data

    def create_rng(self, params):
        return random.Random(params['seed'])

    def create_functions(self, rng, params):
        preparator = NodePreparator(params['weighted'],
                                    params['weight_init'] == 'random',
                                    rng,
                                    params['weight_init_lb'],
                                    params['weight_init_ub'])
        functions = [self.create_function(func_name, preparator, True)
                     for func_name in params['functions'].split(',')]
        return functions

    def create_function(self, function_name: str, prep, cache: bool):
        if function_name == 'Add2':
            return lambda: prep(evo.sr.backpropagation.Add2(cache=cache))
        if function_name == 'Div2':
            return lambda: prep(evo.sr.backpropagation.Div2(cache=cache))
        if function_name == 'Mul2':
            return lambda: prep(evo.sr.backpropagation.Mul2(cache=cache))
        if function_name == 'Sub2':
            return lambda: prep(evo.sr.backpropagation.Sub2(cache=cache))
        if function_name == 'Sin':
            return lambda: prep(evo.sr.backpropagation.Sin(cache=cache))
        if function_name == 'Cos':
            return lambda: prep(evo.sr.backpropagation.Cos(cache=cache))
        if function_name == 'Exp':
            return lambda: prep(evo.sr.backpropagation.Exp(cache=cache))
        if function_name == 'Abs':
            return lambda: prep(evo.sr.backpropagation.Abs(cache=cache))
        if function_name == 'Sqrt':
            return lambda: prep(evo.sr.backpropagation.Sqrt(cache=cache))
        if function_name == 'Sigmoid':
            return lambda: prep(evo.sr.backpropagation.Sigmoid(cache=cache))
        if function_name == 'Tanh':
            return lambda: prep(evo.sr.backpropagation.Tanh(cache=cache))
        if function_name == 'Sinc':
            return lambda: prep(evo.sr.backpropagation.Sinc(cache=cache))
        if function_name == 'Softplus':
            return lambda: prep(evo.sr.backpropagation.Softplus(cache=cache))
        if function_name == 'Gauss':
            return lambda: prep(evo.sr.backpropagation.Gauss(cache=cache))
        if function_name == 'BentIdentity':
            return lambda: prep(evo.sr.backpropagation.BentIdentity(
                cache=cache))

        powmatch = re.fullmatch('Pow\(([0-9]+)\)', function_name)
        if powmatch:
            exponent = int(powmatch.group(1))
            if exponent <= 0:
                raise ValueError('Power in Pow(n) must be a positive integer.')
            return lambda: prep(evo.sr.backpropagation.Power(power=exponent,
                                                             cache=cache))
        raise ValueError('Unrecognized function name {}.'.format(function_name))

    def get_params(self, ns: argparse.Namespace):
        params = dict()

        self.get_logging_params(ns, params)
        self.get_input_params(ns, params)
        self.get_output_params(ns, params)
        self.get_algorithm_params(ns, params)

        return params

    def get_algorithm_params(self, ns, params):
        params['seed'] = ns.seed
        params['generations'] = ns.generations
        params['time'] = ns.time
        params['generation_time_combinator'] = ns.generation_time_combinator
        params['limits'] = {
            'max-genes': ns.max_genes,
            'max-depth': ns.max_depth,
            'max-nodes': ns.max_nodes
        }
        params['pop_size'] = ns.pop_size
        params['tournament_size'] = int(round(ns.pop_size * ns.tournament_size))
        assert params['tournament_size'] > 0, 'Effective tournament size is 0.'
        params['elitism'] = int(round(ns.pop_size * ns.elitism))
        params['bprop_steps_min'] = ns.min_backpropagation_steps
        params['bprop_steps'] = ns.backpropagation_steps
        params['backpropagation_mode'] = ns.backpropagation_mode
        if params['backpropagation_mode'] == 'none':
            params['bprop_steps_min'] = 0
            params['bprop_steps'] = 0
        params['pr_x'] = ns.crossover_prob
        params['pr_hl_x'] = ns.highlevel_crossover_prob
        params['r_hl_x'] = ns.highlevel_crossover_rate
        params['pr_m'] = ns.mutation_prob
        params['pr_c_m'] = ns.constant_mutation_prob
        params['sigma_c_m'] = ns.constant_mutation_sigma
        params['pr_w_m'] = ns.weights_mutation_prob
        params['sigma_w_m'] = ns.weights_mutation_sigma
        params['lcf_mode'] = ns.lcf_mode
        params['weight_init'] = ns.weight_init
        params['const_init_lb'] = min(ns.const_init_bounds)
        params['const_init_ub'] = max(ns.const_init_bounds)
        params['weight_init_lb'] = min(ns.weight_init_bounds)
        params['weight_init_ub'] = max(ns.weight_init_bounds)
        params['weighted'] = ns.weighted
        params['functions'] = ns.functions
        params['constants'] = ns.constants

    def get_output_params(self, ns, params):
        params['output_string_template'] = ns.output_string_template
        params['m_fun'] = ns.m_fun
        params['output_directory'] = ns.output_directory

    def get_input_params(self, ns, params):
        params['training-data'] = ns.training_data
        params['testing-data'] = ns.testing_data
        params['delimiter'] = ns.delimiter

    def get_logging_params(self, ns, params):
        params['logconf'] = ns.logconf

    def log_params(self, params):
        self.log_algorithm_params(params)

    def log_algorithm_params(self, params):
        logging.info('Seed: %d', params['seed'])
        logging.info('Generations limit: %s', params['generations'])
        logging.info('Time limit: %f', params['time'])
        logging.info('Generations + time: %s',
                     params['generation_time_combinator'])
        logging.info('Max genes: %s', params['limits']['max-genes'])
        logging.info('Max depth: %s', params['limits']['max-depth'])
        logging.info('Max nodes: %s', params['limits']['max-nodes'])
        logging.info('Population size: %d', params['pop_size'])
        logging.info('Tournament size: %d', params['tournament_size'])
        logging.info('Elitism: %d', params['elitism'])
        logging.info('Backpropagation mode: %s', params['backpropagation_mode'])
        logging.info('Backpropagation min. steps: %d',
                     params['bprop_steps_min'])
        logging.info('Backpropagation steps: %s', params['bprop_steps'])
        logging.info('Crossover prob.: %f', params['pr_x'])
        logging.info('High-level crossover prob.: %f', params['pr_hl_x'])
        logging.info('High-level crossover rate: %f', params['r_hl_x'])
        logging.info('Mutation prob.: %f', params['pr_m'])
        logging.info('Constant mutation prob.: %f', params['pr_c_m'])
        logging.info('Constant mutation sigma: %f', params['sigma_c_m'])
        logging.info('Weights mutation prob.: %f', params['pr_w_m'])
        logging.info('Weights mutation sigma: %f', params['sigma_w_m'])
        logging.info('LCF mode: %s', params['lcf_mode'])
        logging.info('Weight init: %s', params['weight_init'])
        logging.info('Const init bounds: [%f, %f]', params['const_init_lb'],
                     params['const_init_ub'])
        logging.info('Weight init bounds: [%f, %f]', params['weight_init_lb'],
                     params['weight_init_ub'])
        logging.info('Weighted: %s', params['weighted'])
        logging.info('Functions: %s', params['functions'])
        logging.info('Backpropagation mode: %s', params['backpropagation_mode'])
        logging.info('Generation-time combinator: %s',
                     params['generation_time_combinator'])
        logging.info('Constants mode: %s', params['constants'])

    def create_population_strategy(self, params):
        ps = evo.GenerationalPopulationStrategy(params['pop_size'],
                                                params['elitism'])
        return ps

    def create_mutation(self, rng, functions, terminals, params):
        if ((params['weighted'] or params['lcf_mode'] != 'none') and
                    params['pr_w_m'] > 0):
            if params['lcf_mode'] == 'global':
                muts = [
                    (1 - params['pr_c_m'], evo.gp.SubtreeMutation(
                        float('inf'), rng, functions, terminals,
                        params['limits'])),
                    (params['pr_c_m'], evo.sr.bpgp.ConstantsMutation(
                        params['sigma_c_m'], rng))
                ]
            else:
                muts = [
                    (1 - params['pr_w_m'] - params['pr_c_m'],
                     evo.gp.SubtreeMutation(float('inf'), rng, functions,
                                            terminals, params['limits'])),
                    (params['pr_c_m'], evo.sr.bpgp.ConstantsMutation(
                        params['sigma_c_m'], rng)),
                    (params['pr_w_m'], evo.sr.bpgp.CoefficientsMutation(
                        params['sigma_w_m'], rng))
                ]
            mutation = evo.gp.StochasticChoiceMutation(
                muts, rng, fallback_method=muts[0][1])
        else:
            muts = [
                (1 - params['pr_c_m'], evo.gp.SubtreeMutation(
                    float('inf'), rng, functions, terminals,
                    params['limits'])),
                (params['pr_c_m'], evo.sr.bpgp.ConstantsMutation(
                    params['sigma_c_m'], rng))
            ]
            mutation = evo.gp.StochasticChoiceMutation(
                muts, rng, fallback_method=muts[0][1])
            params['pr_w_m'] = 0

        return mutation

    def create_crossover(self, rng, params):
        if params['limits']['max-genes'] > 1 and params['pr_hl_x'] > 0:
            crossover = evo.gp.StochasticChoiceCrossover([
                (params['pr_hl_x'], evo.gp.CrHighlevelCrossover(
                    params['r_hl_x'], rng, params['limits'])),
                (1 - params['pr_hl_x'], evo.gp.SubtreeCrossover(
                    rng, params['limits']))
            ], rng)
        else:
            crossover = evo.gp.SubtreeCrossover(rng, params['limits'])

        return crossover

    def create_fitness(self, params, x, y):
        if params['backpropagation_mode'] not in ['raw', 'none']:
            steps = (params['backpropagation_mode'], params['bprop_steps'])
        else:
            steps = params['bprop_steps']
        fitness = evo.sr.bpgp.RegressionFitness(
            handled_errors=[],
            train_inputs=x,
            train_output=y,
            updater=evo.sr.backpropagation.IRpropMinus(maximize=True),
            steps=steps,
            min_steps=params['bprop_steps_min'],
            fit=True,
            synchronize_lincomb_vars=params['lcf_mode'] == 'synced',
            ## stats=stats,
            fitness_measure=evo.sr.bpgp.ErrorMeasure.R2,
            backpropagate_only=params['lcf_mode'] == 'global'
        )
        return fitness

    def create_terminals(self, rng, x, cache, params):
        terms = []
        terms += [lambda n=n: evo.sr.Variable(index=n, cache=cache)
                  for n in range(x.shape[1])]
        global_lcs = None
        if params['lcf_mode'] != 'none':
            lc_prep = NodePreparator(
                True, params['lcf_mode'] not in ['synced', 'global'], rng,
                params['weight_init_lb'], params['weight_init_ub'])
            global_lcs = []
            for n in range(x.shape[1]):
                terms.append(
                    lambda n=n: lc_prep(evo.sr.backpropagation.LincombVariable(
                        index=n, num_vars=x.shape[1], cache=cache)))
                global_lcs.append(terms[-1]())
        if params['constants'] == 'classical':
            terms += [
                lambda: evo.sr.Const(rng.uniform(
                    params['const_init_lb'], params['const_init_ub']))
            ]
        elif params['constants'] == 'tunable':
            terms += [
                lambda: evo.sr.backpropagation.TunableConst(
                    rng.uniform(params['const_init_lb'],
                                params['const_init_ub']))
            ]
        return global_lcs, terms

    def create_stopping_condition(self, params):
        if math.isinf(params['time']) and math.isinf(params['generations']):
            logging.warning('Both time and generational stopping condition '
                            'will never be met. Algorithm must be terminated '
                            'externally.')
        time_stop = evo.gp.Gp.time(params['time'])
        generations_stop = evo.gp.Gp.generations(params['generations'])
        if params['generation_time_combinator'] == 'any':
            stop = evo.gp.Gp.any(time_stop, generations_stop)
        elif params['generation_time_combinator'] == 'all':
            stop = evo.gp.Gp.all(time_stop, generations_stop)
        else:
            raise ValueError('Invalid generation-time-combinator')

        return stop

    def create_callback(self, params):
        if params['backpropagation_mode'] != 'none':
            class Cb(evo.gp.Callback):
                def iteration_start(self, algorithm: evo.Evolution):
                    for i in algorithm.population:
                        i.set_fitness(None)

            cb = Cb()
        else:
            cb = None

        return cb

    def create_reproduction_strategy(self, rng, crossover, mutation, functions,
                                     terminals, params):
        return evo.gp.ChoiceReproductionStrategy(
            functions, terminals, rng,
            crossover=crossover,
            mutation=mutation,
            limits=params['limits'],
            crossover_prob=params['pr_x'],
            mutation_prob=params['pr_m'],
            crossover_both=False)

    def create_population_initializer(self, rng, functions, terminals, params):
        return evo.sr.bpgp.FittedForestIndividualInitializer(
            evo.gp.support.RampedHalfHalfInitializer(
                functions=functions,
                terminals=terminals,
                min_depth=1,
                max_depth=params['limits']['max-depth'],
                max_genes=params['limits']['max-genes'],
                generator=rng
            )
        )

    def create_algorithm(self, rng, functions, terminals, global_lcs, fitness,
                         population_strategy, reproduction_strategy, callback,
                         stopping_condition, population_initializer,
                         params: dict):
        # Prepare final algorithm
        if params['lcf_mode'] == 'global':
            alg_class = evo.sr.bpgp.GlobalLincombsGp
            # noinspection PyUnboundLocalVariable
            extra_kwargs = {'global_lcs': global_lcs,
                            'update_steps': params['bprop_steps'],
                            'coeff_mut_prob': params['pr_w_m'],
                            'coeff_mut_sigma': params['sigma_w_m']}
        else:
            alg_class = evo.gp.Gp
            extra_kwargs = {}
        alg = alg_class(
            fitness=fitness,
            pop_strategy=population_strategy,
            selection_strategy=evo.TournamentSelectionStrategy(
                params['tournament_size'], rng),
            reproduction_strategy=reproduction_strategy,
            population_initializer=population_initializer,
            functions=functions,
            terminals=terminals,
            stop=stopping_condition,
            generator=rng,
            limits=params['limits'],
            callback=callback,
            **extra_kwargs
        )
        return alg

    def postprocess(self, algorithm, x_data_trn, y_data_trn, x_data_tst,
                    y_data_tst, output, ns: argparse.Namespace):
        runtime = algorithm.end_time - algorithm.start_time
        bsfs = algorithm.fitness.bsfs
        iterations = algorithm.iterations
        fitness_evals = algorithm.fitness.evaluation_count
        del algorithm

        y_trn = None
        y_tst = None
        cycle = True
        while cycle:
            bsf = bsfs.pop()
            try:
                y_trn = self.eval_individual(x_data_trn, bsf.bsf)
                if x_data_tst is not None:
                    y_tst = self.eval_individual(x_data_tst, bsf.bsf)
                cycle = False
            except:
                logging.exception('Exception during final evaluation.')
                del bsf
                gc.collect()
                logging.error('{} bsfs left'.format(len(bsfs)))
                if len(bsfs) == 0:
                    logging.error(
                        'None of the %f BSFs evaluated without exception.')
                    logging.info('Runtime: {:.3f}'.format(runtime))
                    raise PropagateExit(1)
        r2_trn = r2(y_data_trn, y_trn)
        mse_trn = mse(y_data_trn, y_trn)
        mae_trn = mae(y_data_trn, y_trn)
        wcae_trn = wcae(y_data_trn, y_trn)
        r2_tst = None
        mse_tst = None
        mae_tst = None
        wcae_tst = None
        if y_data_tst is not None and y_tst is not None:
            r2_tst = r2(y_data_tst, y_tst)
            mse_tst = mse(y_data_tst, y_tst)
            mae_tst = mae(y_data_tst, y_tst)
            wcae_tst = wcae(y_data_tst, y_tst)
        nodes = sum(g.get_subtree_size() for g in bsf.bsf.genotype)
        depth = max(g.get_subtree_depth() for g in bsf.bsf.genotype)

        if output['y_trn'] is not None:
            np.savetxt(output['y_trn'], y_trn, delimiter=',')
        if output['y_tst'] is not None and y_tst is not None:
            np.savetxt(output['y_tst'], y_tst, delimiter=',')
        if output['summary'] is not None:
            with open(output['summary'], 'w') as out:
                model_str = evo.sr.bpgp.full_model_str(bsf.bsf,
                                                       num_format='repr',
                                                       newline_genes=True)
                print('model: {}'.format(model_str), file=out)
                print('simplified model: {}'.format(str(bsf.bsf)), file=out)
                print('this model found in iteration: {}'.format(bsf.iteration),
                      file=out)
                print('this model found in fitness evaluation: {}'.format(
                    bsf.eval_count), file=out)
                print('R2 train: {}'.format(r2_trn), file=out)
                if r2_tst is not None:
                    print('R2 test: {}'.format(r2_tst), file=out)
                print('MSE train: {}'.format(mse_trn), file=out)
                if mse_tst is not None:
                    print('MSE test:  {}'.format(mse_tst), file=out)
                print('RMSE train: {}'.format(math.sqrt(mse_trn)), file=out)
                if mse_tst is not None:
                    print('RMSE test:  {}'.format(math.sqrt(mse_tst)), file=out)
                print('MAE train: {}'.format(mae_trn), file=out)
                if mae_tst is not None:
                    print('MAE test:  {}'.format(mae_tst), file=out)
                print('WCAE train: {}'.format(wcae_trn), file=out)
                if mae_tst is not None:
                    print('WCAE test:  {}'.format(wcae_tst), file=out)
                print('nodes: {}'.format(nodes), file=out)
                print('depth: {}'.format(depth), file=out)
                print('time: {}'.format(runtime), file=out)
                print('iterations: {}'.format(iterations), file=out)
                print('fitness evaluations: {}'.format(fitness_evals), file=out)
                print('seed: {}'.format(ns.seed), file=out)
        if output['m_func_templ'] is not None:
            m_fun_fn = output['m_func_templ'].format(output['m_fun'])
            logging.info('Writing matlab function to %s', m_fun_fn)
            with open(m_fun_fn, 'w') as out:
                print(bsf.bsf.to_matlab(output['m_fun']), file=out)
        logging.info('Training R2: {}'.format(r2_trn))
        if r2_tst is not None:
            logging.info('Testing R2: {}'.format(r2_tst))
        logging.info('Runtime: {:.3f}'.format(runtime))
        if output['output_string_template'] is not None:
            output_string = output['output_string_template'].format(
                tst_r2=r2_tst,
                trn_r2=r2_trn,
                tst_r2_inv=1 - r2_tst,
                trn_r2_inv=1 - r2_trn,
                tst_mse=mse_tst,
                trn_mse=mse_trn,
                tst_mae=mae_tst,
                trn_mae=mae_trn,
                tst_wae=wcae_tst,
                trn_wae=wcae_trn,
                runtime=runtime,
                seed=ns.seed,
                iterations=iterations)
            logging.info('Output string: %s', output_string)
            print(output_string)

    def eval_individual(self, x, individual):
        for g in individual.genotype:
            g.clear_cache()

        if individual.genes_num == 1:
            outputs = individual.genotype[0].eval(args=x)
        else:
            outputs = [g.eval(args=x) for g
                       in individual.genotype]
            outputs = evo.utils.column_stack(*outputs)

        intercept = individual.intercept
        coefficients = individual.coefficients
        if coefficients is not None:
            if outputs.ndim == 1 or outputs.ndim == 0:
                outputs = outputs * coefficients
            else:
                outputs = outputs.dot(coefficients)
        if intercept is not None:
            outputs = outputs + intercept

        if outputs.size == 1:
            outputs = np.repeat(outputs, x.shape[0])
        return outputs


def r2(y, yhat):
    ssw = np.sum((y - y.mean()) ** 2)
    e = y - yhat
    return 1 - e.dot(e) / ssw


def mse(y, yhat):
    err = y - yhat
    return err.dot(err) / err.size


def mae(y, yhat):
    err = y - yhat
    return np.sum(np.abs(err)) / err.size


def wcae(y, yhat):
    err = y - yhat
    return np.max(np.abs(err))


class NodePreparator(object):
    def __init__(self, weighted, randomize, rng, lb, ub):
        self.weighted = weighted
        self.randomize = randomize
        self.rng = rng
        self.lb = lb
        self.ub = ub

    def __call__(self, node: evo.sr.backpropagation.WeightedNode):
        if not self.weighted:
            node.tune_bias = False
            node.tune_weights = False

        if not self.randomize:
            return node

        if node.tune_bias is True:
            for i in range(len(node.bias)):
                node.bias[i] = self.rng.uniform(self.lb, self.ub)
        elif node.tune_bias:
            for i in range(len(node.bias)):
                if node.tune_bias[i]:
                    node.bias[i] = self.rng.uniform(self.lb, self.ub)
        if node.tune_weights is True:
            for i in range(len(node.weights)):
                node.weights[i] = self.rng.uniform(self.lb, self.ub)
        elif node.tune_weights:
            for i in range(len(node.weights)):
                if node.tune_weights[i]:
                    node.weights[i] = self.rng.uniform(self.lb, self.ub)
        return node
