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
import evo.sr
import evo.sr.backpropagation
import evo.sr.bpgp
import evo.sr.gp
import evo.utils
from evo.runners import text, bounded_integer, bounded_float, float01, \
    DataSpec, PropagateExit


class Runner(object):
    PARSER_ARG = 'gp'

    def __init__(self, subparsers):
        self.parser = subparsers.add_parser(
            self.PARSER_ARG, help='Genetic Programming',
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
            '--training-data-skiprows',
            help=text(
                'Determines how many rows at the top of the training data file '
                'should be skipped before loading the actual data. Default is '
                '0.'),
            type=int,
            required=False,
            default=0)
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
            '--testing-data-skiprows',
            help=text(
                'Determines how many rows at the top of the testing data file '
                'should be skipped before loading the actual data. Default is '
                '0.'),
            type=int,
            required=False,
            default=0)
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
            '--m-fun-each',
            help=text(
                'If specified, a matlab function for each BSF acquired will be '
                'written. In this case a {iteration} placeholder will be '
                'replaced by the iteration number of the particular BSF. If '
                'not specified, only the last BSF will be written out and no '
                'placeholder will be replaced.'
            ),
            action='store_true')
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
                'Tanh, Sinc, Softplus, Gauss, BentIdentity, Signum, '
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
                'of this parameter is then the probability of subtree '
                'mutation. To turn this mutation off, set the parameter to 0. '
                'Default is 0.05.'),
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
            '--const-init-bounds',
            help=text(
                'Bounds of the range the constants (leaf '
                'nodes) are sampled from. Default is -10 and '
                '10.'),
            nargs=2,
            type=float,
            metavar='bound',
            default=[-10, 10])

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
        if (x_data_tst is not None and
                x_data_trn.shape[1] != x_data_tst.shape[1]):
            logging.error('Training and testing data have different number of '
                          'columns. Exitting.')
            raise PropagateExit(1)
        output = self.prepare_output(params)

        self.log_params(params)
        # prepare stuff needed for algorithm creation
        rng = self.create_rng(params)
        functions = self.create_functions(params)
        terminals = self.create_terminals(rng, x_data_trn, True,
                                                      params)
        fitness = self.create_fitness(params, x_data_trn, y_data_trn)
        crossover = self.create_crossover(rng, params)
        mutation = self.create_mutation(rng, functions, terminals, params)
        population_strategy = self.create_population_strategy(params)
        reproduction_strategy = self.create_reproduction_strategy(
            rng, crossover, mutation, functions, terminals, params)
        # noinspection PyNoneFunctionAssignment
        callback = self.create_callback(params)
        stopping_condition = self.create_stopping_condition(params)
        population_initializer = self.create_population_initializer(rng,
                                                                    functions,
                                                                    terminals,
                                                                    params)
        # create algorithm
        algorithm = self.create_algorithm(rng, functions, terminals, fitness,
                                          population_strategy,
                                          reproduction_strategy, callback,
                                          stopping_condition,
                                          population_initializer, params)

        # set numpy to raise an error for everything except underflow
        np.seterr(all='raise', under='warn')
        result = algorithm.run()
        self.postprocess(algorithm, x_data_trn, y_data_trn, x_data_tst,
                         y_data_tst, output, params)
        if result:
            return 0
        return 100

    def load_data(self, params: dict):
        def load(ds: DataSpec, delimiter: str, skiprows: int, prefix: str):
            if not os.path.isfile(ds.file):
                logging.error('%s data file %s does not exist or is not a '
                              'file. Exitting.', prefix, ds.file)
                raise PropagateExit(1)
            logging.info('%s data file: %s', prefix, ds.file)
            data = np.loadtxt(ds.file, delimiter=delimiter, skiprows=skiprows)
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
        training_sk = params['training-data-skiprows']
        testing_ds = params['testing-data']
        testing_sk = params['testing-data-skiprows']

        training_x, training_y = load(training_ds, dlm, training_sk, 'Training')
        logging.info('Training X data shape (rows, cols): %s', training_x.shape)
        logging.info('Training Y data shape (elements,): %s', training_y.shape)
        testing_x, testing_y = None, None
        if testing_ds is not None:
            testing_x, testing_y = load(testing_ds, dlm, testing_sk, 'Testing')
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

    def create_functions(self, params):
        functions = [self.create_function(func_name, True)
                     for func_name in params['functions'].split(',')]
        return functions

    def create_function(self, function_name: str, cache: bool):
        if function_name == 'Add2':
            return lambda: evo.sr.Add2(cache=cache)
        if function_name == 'Div2':
            return lambda: evo.sr.Div2(cache=cache)
        if function_name == 'Mul2':
            return lambda: evo.sr.Mul2(cache=cache)
        if function_name == 'Sub2':
            return lambda: evo.sr.Sub2(cache=cache)
        if function_name == 'Sin':
            return lambda: evo.sr.Sin(cache=cache)
        if function_name == 'Cos':
            return lambda: evo.sr.Cos(cache=cache)
        if function_name == 'Exp':
            return lambda: evo.sr.Exp(cache=cache)
        if function_name == 'Abs':
            return lambda: evo.sr.Abs(cache=cache)
        if function_name == 'Sqrt':
            return lambda: evo.sr.Sqrt(cache=cache)
        if function_name == 'Sigmoid':
            return lambda: evo.sr.Sigmoid(cache=cache)
        if function_name == 'Tanh':
            return lambda: evo.sr.Tanh(cache=cache)
        if function_name == 'Sinc':
            return lambda: evo.sr.Sinc(cache=cache)
        if function_name == 'Softplus':
            return lambda: evo.sr.Softplus(cache=cache)
        if function_name == 'Gauss':
            return lambda: evo.sr.Gauss(cache=cache)
        if function_name == 'BentIdentity':
            return lambda: evo.sr.BentIdentity(cache=cache)
        if function_name == 'Signum':
            return lambda: evo.sr.Signum(cache=cache)

        powmatch = re.fullmatch('Pow\(([0-9]+)\)', function_name)
        if powmatch:
            exponent = int(powmatch.group(1))
            if exponent <= 0:
                raise ValueError('Power in Pow(n) must be a positive integer.')
            return lambda: evo.sr.Power(power=exponent, cache=cache)
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
            'max-depth': ns.max_depth,
            'max-nodes': ns.max_nodes
        }
        params['pop_size'] = ns.pop_size
        params['tournament_size'] = int(round(ns.pop_size * ns.tournament_size))
        assert params['tournament_size'] > 0, 'Effective tournament size is 0.'
        params['elitism'] = int(round(ns.pop_size * ns.elitism))
        params['pr_x'] = ns.crossover_prob
        params['pr_m'] = ns.mutation_prob
        params['pr_c_m'] = ns.constant_mutation_prob
        params['sigma_c_m'] = ns.constant_mutation_sigma
        params['const_init_lb'] = min(ns.const_init_bounds)
        params['const_init_ub'] = max(ns.const_init_bounds)
        params['functions'] = ns.functions

    def get_output_params(self, ns, params):
        params['output_string_template'] = ns.output_string_template
        params['m_fun'] = ns.m_fun
        params['m_fun_each'] = ns.m_fun_each
        params['output_directory'] = ns.output_directory

    def get_input_params(self, ns, params):
        params['training-data'] = ns.training_data
        params['training-data-skiprows'] = ns.training_data_skiprows
        params['testing-data'] = ns.testing_data
        params['testing-data-skiprows'] = ns.testing_data_skiprows
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
        logging.info('Max depth: %s', params['limits']['max-depth'])
        logging.info('Max nodes: %s', params['limits']['max-nodes'])
        logging.info('Population size: %d', params['pop_size'])
        logging.info('Tournament size: %d', params['tournament_size'])
        logging.info('Elitism: %d', params['elitism'])
        logging.info('Crossover prob.: %f', params['pr_x'])
        logging.info('Mutation prob.: %f', params['pr_m'])
        logging.info('Constant mutation prob.: %f', params['pr_c_m'])
        logging.info('Constant mutation sigma: %f', params['sigma_c_m'])
        logging.info('Const init bounds: [%f, %f]', params['const_init_lb'],
                     params['const_init_ub'])
        logging.info('Functions: %s', params['functions'])
        logging.info('Generation-time combinator: %s',
                     params['generation_time_combinator'])

    def create_population_strategy(self, params):
        ps = evo.GenerationalPopulationStrategy(params['pop_size'],
                                                params['elitism'])
        return ps

    def create_mutation(self, rng, functions, terminals, params):
        muts = [(1 - params['pr_c_m'],
                 evo.gp.SubtreeMutation(float('inf'), rng, functions, terminals,
                                        params['limits'])),
                (params['pr_c_m'],
                 evo.sr.ConstantsMutation(params['sigma_c_m'], rng))]
        mutation = evo.gp.StochasticChoiceMutation(
            muts, rng, fallback_method=muts[0][1])

        return mutation

    def create_crossover(self, rng, params):
        crossover = evo.gp.SubtreeCrossover(rng, params['limits'])

        return crossover

    def create_fitness(self, params, x, y):
        fitness = evo.sr.gp.RegressionFitness(
            train_inputs=x,
            train_output=y,
            error_fitness=evo.sr.ErrorMeasure.R2.worst,
            handled_errors=[OverflowError, ZeroDivisionError,
                            FloatingPointError],
            fitness_measure=evo.sr.ErrorMeasure.R2
        )
        return fitness

    def create_terminals(self, rng, x, cache, params):
        terms = []
        terms += [lambda n=n: evo.sr.Variable(index=n, cache=cache)
                  for n in range(x.shape[1])]
        terms += [lambda: evo.sr.Const(rng.uniform(params['const_init_lb'],
                                                   params['const_init_ub']))]
        return terms

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
        return evo.gp.support.RampedHalfHalfInitializer(
            functions=functions,
            terminals=terminals,
            min_depth=1,
            max_depth=params['limits']['max-depth'],
            max_genes=1,
            generator=rng
        )

    def create_algorithm(self, rng, functions, terminals, fitness,
                         population_strategy, reproduction_strategy, callback,
                         stopping_condition, population_initializer,
                         params: dict):
        # Prepare final algorithm
        alg = evo.gp.Gp(
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
            callback=callback
        )
        return alg

    def postprocess(self, algorithm, x_data_trn, y_data_trn, x_data_tst,
                    y_data_tst, output, params):
        runtime = algorithm.end_time - algorithm.start_time
        bsfs = algorithm.fitness.bsfs
        iterations = algorithm.iterations
        fitness_evals = algorithm.fitness.evaluation_count
        del algorithm

        if params['m_fun_each'] and output['m_func_templ'] is not None:
            for bsf in bsfs:
                m_fun_fn = output['m_func_templ'].format(
                    output['m_fun'].format(iteration=bsf.iteration))
                logging.info('Writing matlab function to %s', m_fun_fn)
                comments = ['Iteration: ' + str(bsf.iteration),
                            'Fitness: ' + str(bsf.bsf.get_fitness())]
                with open(m_fun_fn, 'w') as out:
                    print(bsf.bsf.to_matlab(
                        output['m_fun'].format(iteration=bsf.iteration),
                        comments), file=out)

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
            except BaseException as e:
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
                model_str = evo.sr.gp.full_model_str(bsf.bsf,
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
                print('seed: {}'.format(params['seed']), file=out)
        if output['m_func_templ'] is not None and not params['m_fun_each']:
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
                seed=params['seed'],
                iterations=iterations)
            logging.info('Output string: %s', output_string)
            print(output_string)

    def eval_individual(self, x, individual):
        individual.genotype[0].clear_cache()

        return individual.genotype[0].eval(args=x)


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
