# -*- coding: utf-8 -*-
import argparse
import collections
import gc
import logging
import logging.config
import math
import os
import random
import sys
import time

import numpy as np
import yaml
from pkg_resources import resource_stream

import evo.gp
import evo.sr.backpropagation
import evo.sr.bpgp
import evo.utils
from evo.runners import text, bounded_integer, bounded_float, float01, DataSpec


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
    parser.add_argument('--training-data',
                        help=text('Specification of the training data in the '
                                  'format file[:x-columns:y-column].\n\n'
                                  '"file" is a path to a CSV file to be '
                                  'loaded.\n\n'
                                  '"x-columns" is a comma-separated '
                                  '(only comma, without whitespaces) list of '
                                  'numbers of columns (zero-based) to be used '
                                  'as the features.\n\n'
                                  '"y-column" is a number of column that '
                                  'will be used as the target. You can use '
                                  'negative numbers to count from back, i.e. '
                                  '-1 is the last column, -2 is the second to '
                                  'the last column, etc.\n\n'
                                  'The bracketed part is optional and can be '
                                  'left out. If it is left out, all columns '
                                  'except the last one are used as features '
                                  'and the last one is used as target.'),
                        type=DataSpec,
                        required=True,
                        metavar='file[:x-columns:y-column]')
    parser.add_argument('--testing-data',
                        help=text('Specification of the testing data. The '
                                  'format is identical to the one of '
                                  '--training-data. The testing data must have '
                                  'the same number of columns as'
                                  '--training-data.\n\n'
                                  'The testing data are evaluated with the '
                                  'best individual after the evolution '
                                  'finishes. If, for some reason, the '
                                  'individual fails to evaluate (e.g. due to '
                                  'numerical errors like division by zero), '
                                  'the second best individual is tried and so '
                                  'forth until the individual evaluates or '
                                  'there is no individual left.\n\n'
                                  'If the testing data is not specified, no '
                                  'testing is done (only training measures are '
                                  'reported).'),
                        type=DataSpec,
                        required=False,
                        metavar='file[:x-columns:y-column]')
    parser.add_argument('--delimiter',
                        help=text('Field delimiter of the CSV files specified '
                                  'in --training-data and --testing-data. '
                                  'Default is ",".'),
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
                                  'for. Default is infinity (i.e. until '
                                  'stopped externally or with some other '
                                  'stopping condition).'),
                        type=bounded_integer(1),
                        default=float('inf'))
    parser.add_argument('--time',
                        help=text('The maximum number of seconds to run for. '
                                  'Default is infinity (i.e. until stopped '
                                  'externally or with some other stopping '
                                  'condition).'),
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
                        help=text('Population size. Default is 100.'),
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
                                  'subtree crossover. If --max-genes is 1, '
                                  'this parameter is ignored (even if not '
                                  'specified) and set to 0. Default is 0.2.'),
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
                        help=text('How backpropagation is used. '
                                  'Mode "none" turns the backpropagation off '
                                  'completely. Mode "raw" means that the '
                                  'number of steps is always the number '
                                  'specified in --backpropagation-steps (and '
                                  'hence --min-backpropagation-steps is '
                                  'ignored). Modes "nodes" and "depth" mean '
                                  'that the number of steps is the number '
                                  'specified in --backpropagation-steps minus '
                                  'the total number of nodes of the individual '
                                  '(for "nodes") or the maximum depth of the '
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
                        help=text('At least this number of backpropagation '
                                  'steps is always performed, no matter what '
                                  '--backpropagation-steps and '
                                  '--backpropagation-mode are set to (except '
                                  'for "none" mode). Default is 2.'),
                        type=bounded_integer(0),
                        default=2)
    parser.add_argument('--weighted',
                        help=text('If specified, the inner nodes will be '
                                  'weighted, i.e. with multiplicative and '
                                  'additive weights, tunable by '
                                  'backpropagation and weights mutation.'),
                        action='store_true')
    parser.add_argument('--lcf-mode',
                        help=text('How the LCFs are used. '
                                  'Mode "none" turns the LCFs off completely. '
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
    parser.add_argument('--weight-init-bounds',
                        help=text('Bounds of the range the weights are sampled '
                                  'from when --weight-init is set to "random". '
                                  'Default is -10 and 10.'),
                        nargs=2,
                        type=float,
                        metavar='bound',
                        default=[-10, 10])
    parser.add_argument('--const-init-bounds',
                        help=text('Bounds of the range the constants (leaf '
                                  'nodes) are sampled from. Default is -10 and '
                                  '10.'),
                        nargs=2,
                        type=float,
                        metavar='bound',
                        default=[-10, 10])


# noinspection PyUnresolvedReferences
def handle(ns: argparse.Namespace):
    with resource_stream('evo.resources', 'logging-default.yaml') as f:
        logging_conf = yaml.load(f)
    if ns.logconf is not None:
        if not os.path.isfile(ns.logconf):
            logging.config.dictConfig(logging_conf)
            logging.error('Supplied logging configuration file does not exist '
                          'or is not a file. Exitting.')
            sys.exit(1)
        with open(ns.logconf) as f:
            local = yaml.load(f)
        evo.utils.nested_update(logging_conf, local)
    logging.config.dictConfig(logging_conf)
    logging.info('Starting evo.')

    x_data_trn, y_data_trn = load_data(ns.training_data, ns.delimiter)
    x_data_tst, y_data_tst = load_data(ns.testing_data, ns.delimiter, True)
    if x_data_tst is not None and x_data_trn.shape[1] != x_data_tst.shape[1]:
        logging.error('Training and testing data have different number of '
                      'columns. Exitting.')
        sys.exit(1)

    output = prepare_output(ns)
    algorithm = create(x=x_data_trn, y=y_data_trn, ns=ns)
    result = algorithm.run()
    postprocess(algorithm, x_data_trn, y_data_trn, x_data_tst, y_data_tst,
                output)


def load_data(ds: DataSpec, delimiter: str, testing: bool=False):
    if ds is None:
        return None, None
    prefix = 'Training'
    if testing:
        prefix = 'Testing'

    if not os.path.isfile(ds.file):
        print('File {} does not exist or is not a file. '
              'Exitting.'.format(ds.file), file=sys.stderr)
        sys.exit(1)
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

    logging.info('%s X data shape (rows, cols): %s', prefix, x_data.shape)
    logging.info('%s Y data shape (elements,): %s', prefix, y_data.shape)

    return x_data, y_data


# noinspection PyUnresolvedReferences
def prepare_output(ns: argparse.Namespace):
    output_data = collections.defaultdict()
    if ns.output_directory is None:
        return output_data

    if os.path.isdir(ns.output_directory):
        logging.warning('Output directory %s already exists! Contents might be '
                        'overwritten', ns.output_directory)
    os.makedirs(ns.output_directory, exist_ok=True)
    logging.info('Output directory (relative): %s',
                 os.path.relpath(ns.output_directory, os.getcwd()))
    logging.info('Output directory (absolute): %s',
                 os.path.abspath(ns.output_directory))

    output_data['y_trn'] = os.path.join(ns.output_directory, 'y_trn.txt')
    output_data['y_tst'] = os.path.join(ns.output_directory, 'y_tst.txt')
    output_data['summary'] = os.path.join(ns.output_directory, 'summary.txt')
    output_data['m_func_templ'] = os.path.join(ns.output_directory, '{}.m')
    output_data['stats'] = os.path.join(ns.output_directory, 'stats.csv')
    return output_data


def create(x, y, ns: argparse.Namespace):
    # Extract simple parameters
    rng = random.Random(ns.seed)
    logging.info('Seed: %d', ns.seed)

    generations = ns.generations
    logging.info('Generations limit: %s', generations)

    t = ns.time
    logging.info('Time limit: %f', t)

    combinator = ns.generation_time_combinator
    logging.info('Generations + time: %s', combinator)

    limits = {
        'max-genes': ns.max_genes,
        'max-depth': ns.max_depth,
        'max-nodes': ns.max_nodes
    }
    logging.info('Max genes: %s', limits['max-genes'])
    logging.info('Max depth: %s', limits['max-depth'])
    logging.info('Max nodes: %s', limits['max-nodes'])

    pop_size = ns.pop_size
    logging.info('Population size: %d', pop_size)

    tournament_size = int(round(pop_size * ns.tournament_size))
    assert tournament_size > 0, 'Effective tournament size is 0.'
    logging.info('Tournament size: %d', tournament_size)

    elitism = int(round(pop_size * ns.elitism))
    logging.info('Elitism: %d', elitism)

    bprop_steps_min = ns.min_backpropagation_steps
    bprop_steps = ns.backpropagation_steps
    logging.info('Backpropagation mode: %s', ns.backpropagation_mode)
    logging.info('Backpropagation min. steps: %d', bprop_steps_min)
    logging.info('Backpropagation steps: %s', bprop_steps)
    if ns.backpropagation_mode == 'none':
        bprop_steps_min = 0
        bprop_steps = 0
    elif ns.backpropagation_mode != 'raw':
        bprop_steps = (ns.backpropagation_mode, bprop_steps)

    pr_x = ns.crossover_prob
    logging.info('Crossover prob.: %f', pr_x)

    pr_hl_x = ns.highlevel_crossover_prob
    logging.info('High-level crossover prob.: %f', pr_hl_x)

    r_hl_x = ns.highlevel_crossover_rate
    logging.info('High-level crossover rate: %f', r_hl_x)

    pr_m = ns.mutation_prob
    logging.info('Mutation prob.: %f', pr_m)

    pr_c_m = ns.constant_mutation_prob
    logging.info('Constant mutation prob.: %f', pr_c_m)

    sigma_c_m = ns.constant_mutation_sigma
    logging.info('Constant mutation sigma: %f', sigma_c_m)

    pr_w_m = ns.weights_mutation_prob
    logging.info('Weights mutation prob.: %f', pr_w_m)

    sigma_w_m = ns.weights_mutation_sigma
    logging.info('Weights mutation sigma: %f', sigma_w_m)

    # Prepare functions and terminals
    logging.info('LCF mode: %s', ns.lcf_mode)
    logging.info('Weight init: %s', ns.weight_init)
    const_init_lb = min(ns.const_init_bounds)
    const_init_ub = max(ns.const_init_bounds)
    logging.info('Const init bounds: [%f, %f]', const_init_lb, const_init_ub)
    weight_init_lb = min(ns.weight_init_bounds)
    weight_init_ub = max(ns.weight_init_bounds)
    logging.info('Weight init bounds: [%f, %f]', weight_init_lb, weight_init_ub)
    prep = NodePreparator(ns.weighted, ns.weight_init == 'random', rng,
                          weight_init_lb,
                          weight_init_ub)
    cache = True
    funcs = [lambda: prep(evo.sr.backpropagation.Add2(cache=cache)),
             # lambda: rnd(evo.sr.backpropagation.Div2(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Mul2(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Sub2(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Sin(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Cos(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Exp(cache=cache)),
             # lambda: rnd(evo.sr.backpropagation.Abs(cache=cache)),
             # lambda: rnd(evo.sr.backpropagation.Sqrt(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Sigmoid(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Tanh(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Sinc(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Softplus(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Gauss(cache=cache)),
             lambda: prep(evo.sr.backpropagation.Power(power=2, cache=cache)),
             lambda: prep(evo.sr.backpropagation.Power(power=3, cache=cache)),
             lambda: prep(evo.sr.backpropagation.Power(power=4, cache=cache)),
             lambda: prep(evo.sr.backpropagation.Power(power=5, cache=cache)),
             lambda: prep(evo.sr.backpropagation.Power(power=6, cache=cache)),
             ]

    terms = []
    terms += [lambda n=n: evo.sr.Variable(index=n, cache=cache)
              for n in range(x.shape[1])]
    if ns.lcf_mode != 'none':
        lc_prep = NodePreparator(True,
                                 ns.lcf_mode not in ['synced', 'global'],
                                 rng,
                                 weight_init_lb,
                                 weight_init_ub)
        global_lcs = []
        for n in range(x.shape[1]):
            terms.append(
                lambda n=n: lc_prep(evo.sr.backpropagation.LincombVariable(
                    index=n, num_vars=x.shape[1], cache=cache)))
            global_lcs.append(terms[-1]())
    terms += [lambda: evo.sr.Const(rng.uniform(const_init_lb, const_init_ub))]

    # Prepare fitness
    fitness = evo.sr.bpgp.RegressionFitness(
        handled_errors=[],
        train_inputs=x,
        train_output=y,
        updater=evo.sr.backpropagation.IRpropMinus(maximize=True),
        steps=bprop_steps,
        min_steps=bprop_steps_min,
        fit=True,
        synchronize_lincomb_vars=ns.lcf_mode == 'synced',
        ## stats=stats,
        fitness_measure=evo.sr.bpgp.ErrorMeasure.R2,
        backpropagate_only=ns.lcf_mode == 'global'
    )

    # Prepare population strategy
    ps = evo.GenerationalPopulationStrategy(pop_size, elitism)

    # Prepare crossover
    if limits['max-genes'] > 1 and ns.highlevel_crossover_prob > 0:
        crossover = evo.gp.StochasticChoiceCrossover([
            (pr_hl_x, evo.gp.CrHighlevelCrossover(r_hl_x, rng, limits)),
            (1 - pr_hl_x, evo.gp.SubtreeCrossover(rng, limits))
        ], rng)
    else:
        crossover = evo.gp.SubtreeCrossover(rng, limits)

    # Prepare mutation
    if (ns.weighted or ns.lcf_mode != 'none') and pr_w_m > 0:
        if ns.lcf_mode == 'global':
            muts = [
                (1 - pr_c_m, evo.gp.SubtreeMutation(float('inf'), rng, funcs,
                                                    terms, limits)),
                (pr_c_m, evo.sr.bpgp.ConstantsMutation(sigma_c_m, rng))
            ]
        else:
            muts = [
                (1 - pr_w_m - pr_c_m, evo.gp.SubtreeMutation(float('inf'), rng,
                                                             funcs, terms,
                                                             limits)),
                (pr_c_m, evo.sr.bpgp.ConstantsMutation(sigma_c_m, rng)),
                (pr_w_m, evo.sr.bpgp.CoefficientsMutation(sigma_w_m, rng))
            ]
        mutation = evo.gp.StochasticChoiceMutation(muts, rng,
                                                   fallback_method=muts[0][1])
    else:
        muts = [
            (1 - pr_c_m, evo.gp.SubtreeMutation(float('inf'), rng, funcs, terms,
                                                limits)),
            (pr_c_m, evo.sr.bpgp.ConstantsMutation(sigma_c_m, rng))
        ]
        mutation = evo.gp.StochasticChoiceMutation(muts, rng,
                                                   fallback_method=muts[0][1])
        pr_w_m = 0

    # Prepare reproduction strategy
    rs = evo.gp.ChoiceReproductionStrategy(
        funcs, terms, rng,
        crossover=crossover,
        mutation=mutation,
        limits=limits,
        crossover_prob=pr_x,
        mutation_prob=pr_m,
        crossover_both=False)

    # Prepare callback
    if ns.backpropagation_mode != 'none':
        def cb(a, situation):
            if situation == evo.gp.Gp.CallbackSituation.iteration_start:
                for i in a.population:
                    i.set_fitness(None)
    else:
        cb = None

    # Prepare stopping condition
    if math.isinf(t) and math.isinf(generations):
        logging.warning('Both time and generational stopping condition will '
                        'never be met. Algorithm must be terminated '
                        'externally.')
    time_stop = evo.gp.Gp.time(t)
    generations_stop = evo.gp.Gp.generations(generations)
    if ns.generation_time_combinator == 'any':
        stop = evo.gp.Gp.any(time_stop, generations_stop)
    elif ns.generation_time_combinator == 'all':
        stop = evo.gp.Gp.all(time_stop, generations_stop)
    else:
        raise ValueError('Invalid generation-time-combinator')

    # Prepare final algorithm
    if ns.lcf_mode == 'global':
        alg_class = evo.sr.bpgp.GlobalLincombsGp
        # noinspection PyUnboundLocalVariable
        extra_kwargs = {'global_lcs': global_lcs,
                        'update_steps': bprop_steps,
                        'coeff_mut_prob': pr_w_m,
                        'coeff_mut_sigma': sigma_w_m}
    else:
        alg_class = evo.gp.Gp
        extra_kwargs = {}
    alg = alg_class(
        fitness=fitness,
        pop_strategy=ps,
        selection_strategy=evo.TournamentSelectionStrategy(tournament_size,
                                                           rng),
        reproduction_strategy=rs,
        population_initializer=evo.sr.bpgp.FittedForestIndividualInitializer(
            evo.gp.support.RampedHalfHalfInitializer(
                functions=funcs,
                terminals=terms,
                min_depth=1,
                max_depth=limits['max-depth'],
                max_genes=limits['max-genes'],
                generator=rng
            )
        ),
        functions=funcs,
        terminals=terms,
        stop=stop,
        generator=rng,
        limits=limits,
        callback=cb,
        **extra_kwargs
    )
    return alg


def postprocess(algorithm, x_data_trn, y_data_trn, x_data_tst, y_data_tst,
                output):
    runtime = algorithm.end_time - algorithm.start_time
    bsfs = algorithm.fitness.bsfs
    del algorithm

    y_trn = None
    y_tst = None
    cycle = True
    while cycle:
        bsf = bsfs.pop()
        try:
            y_trn = eval_individual(x_data_trn, bsf)
            if x_data_tst is not None:
                y_tst = eval_individual(x_data_tst, bsf)
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
                return 1
    r2_trn = r2(y_data_trn, y_trn)
    mse_trn = mse(y_data_trn, y_trn)
    mae_trn = mae(y_data_trn, y_trn)
    r2_tst = None
    mse_tst = None
    mae_tst = None
    if y_data_tst is not None and y_tst is not None:
        r2_tst = r2(y_data_tst, y_tst)
        mse_tst = mse(y_data_tst, y_tst)
        mae_tst = mae(y_data_tst, y_tst)
    nodes = sum(g.get_subtree_size() for g in bsf.genotype)
    depth = max(g.get_subtree_depth() for g in bsf.genotype)

    if output['y_trn'] is not None:
        np.savetxt(output['y_trn'], y_trn, delimiter=',')
    if output['y_tst'] is not None and y_tst is not None:
        np.savetxt(output['y_tst'], y_tst, delimiter=',')
    if output['summary'] is not None:
        with open(output['summary'], 'w') as out:
            print('model: {}'.format(
                evo.sr.bpgp.full_model_str(bsf, num_format='repr')), file=out)
            print('simplified model: {}'.format(str(bsf)), file=out)
            print('time: {}'.format(runtime), file=out)
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
            print('MAE test:  {}'.format(mae_tst), file=out)
            print('nodes: {}'.format(nodes), file=out)
            print('depth: {}'.format(depth), file=out)
    if output['m_func_templ'] is not None:
        with open(output['m_func_templ'].format('_func'), 'w') as out:
            print(bsf.to_matlab('_func'), file=out)
    logging.info('Training R2: {}'.format(r2_trn))
    if r2_tst is not None:
        logging.info('Testing R2: {}'.format(r2_tst))
    logging.info('Runtime: {:.3f}'.format(runtime))


def eval_individual(x, individual):
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
