# -*- coding: utf8 -*-
"""TODO
"""

import copy
import enum
import itertools
import logging
import operator
import textwrap

import numpy

import evo
import evo.gp
import evo.gp.support
import evo.sr
import evo.sr.backpropagation
import evo.utils
import evo.utils.stats


class FittedForestIndividual(evo.gp.support.ForestIndividual):
    def __init__(self, genotype: list, intercept, coefficients):
        super().__init__(genotype)
        self.intercept = intercept
        self.coefficients = coefficients

    def __str__(self):
        if hasattr(self, 'str'):
            return str(self.str)
        return '{interc} + {coeffs} * {genes}'.format(
            interc=self.intercept,
            coeffs=str(self.coefficients).replace('\n', ''),
            genes=str([str(g) for g in self.genotype]))

    def copy(self, carry_evaluation=True, carry_data=True):
        cg = [g.clone() for g in self.genotype]
        clone = FittedForestIndividual(cg,
                                       copy.deepcopy(self.intercept),
                                       copy.deepcopy(self.coefficients))
        evo.Individual.copy_evaluation(self, clone, carry_evaluation)
        evo.Individual.copy_data(self, clone, carry_data)
        return clone

    def to_matlab(self, function_name='_bp_mggp_fn'):
        coeff_mat_str = '; '.join([repr(c) for c in self.coefficients])
        coeff_mat_str = '[' + coeff_mat_str + ']'
        intercept_str = repr(self.intercept)

        gene_exprs = '\n'.join(
            ['g{{{}}} = {};'.format(i + 1, g.to_matlab_expr())
             for i, g in enumerate(self.genotype)])
        genes_str = '[' + ' '.join(['g{}'.format(i)
                                    for i in range(len(self.genotype))]) + ']'

        all_funcs = set()
        for g in self.genotype:
            for n in g.get_nodes_bfs():
                f = n.to_matlab_def()
                if f is not None:
                    all_funcs.add(f)
        all_funcs = sorted(all_funcs)

        matlab_str = textwrap.dedent('''
        function out = {fname}(X)

        intercept = {intercept};
        coefficients = {coeffs};
        g = cell({num_bases}, 1);
        {gene_exprs}
        lengths = unique(cellfun(@(v)(length(v)), g));
        genes = zeros(max(lengths), {num_bases});
        for i = 1:{num_bases}
            genes(:, i) = g{{i}};
        end

        out = intercept + genes * coefficients;

        end
        ''').format(fname=function_name,
                    intercept=intercept_str,
                    coeffs=coeff_mat_str,
                    num_bases=self.genes_num,
                    gene_exprs=gene_exprs).strip()

        return '\n\n'.join([matlab_str] + all_funcs)


class FittedForestIndividualInitializer(evo.PopulationInitializer):
    def __init__(self, other: evo.PopulationInitializer):
        self.other = other

    def initialize(self, pop_size: int, limits: dict):
        pop = self.other.initialize(pop_size, limits)
        return [FittedForestIndividual(
            i.genotype, 0.0, numpy.ones((len(i.genotype),)))
                for i in pop]


class CoefficientsMutation(evo.gp.MutationOperator):
    def __init__(self, sigma, generator):
        self.sigma = sigma
        self.generator = generator

    def mutate(self, i):
        all_nodes = []

        for g in i.genotype:
            all_nodes.extend(g.get_nodes_dfs(
                predicate=CoefficientsMutation.predicate))
        if all_nodes:
            self.mutate_node(self.generator.choice(all_nodes), self.sigma)
            i.set_fitness(None)
        else:
            raise evo.gp.OperatorNotApplicableError('No weighted nodes found.')
        return i

    @staticmethod
    def predicate(n):
        return (isinstance(n, evo.sr.backpropagation.WeightedNode) and
                (n.tune_bias is not False or n.tune_weights is not False))

    def mutate_node(self, node, sigma):
        if node.tune_bias is True:
            for i in range(node.bias.size):
                node.bias[i] += self.generator.gauss(0, sigma)
        elif node.tune_bias:
            for i in range(node.bias.size):
                if node.tune_bias[i]:
                    node.bias[i] += self.generator.gauss(0, sigma)

        if node.tune_weights is True:
            for i in range(node.weights.size):
                node.weights[i] += self.generator.gauss(0, sigma)
        elif node.tune_weights:
            for i in range(node.weights.size):
                if node.tune_weights[i]:
                    node.weights[i] += self.generator.gauss(0, sigma)

        node.notify_change()


class ConstantsMutation(evo.gp.MutationOperator):
    def __init__(self, sigma, generator):
        self.sigma = sigma
        self.generator = generator

    def mutate(self, i):
        all_nodes = []

        for g in i.genotype:
            all_nodes.extend(g.get_nodes_dfs(
                predicate=ConstantsMutation.predicate))
        if all_nodes:
            self.mutate_node(self.generator.choice(all_nodes), self.sigma)
            i.set_fitness(None)
        else:
            raise evo.gp.OperatorNotApplicableError('No constants found.')
        return i

    @staticmethod
    def predicate(n):
        return isinstance(n, evo.sr.Const)

    def mutate_node(self, node, sigma):
        node.value += self.generator.gauss(0, sigma)
        node.data['name'] = str(node.value)

        node.notify_change()


class ErrorMeasure(enum.Enum):
    R2 = 0
    MSE = 1
    MAE = 2
    WORST_CASE_AE = 3

    @property
    def worst(self):
        if self is ErrorMeasure.R2:
            return -numpy.inf
        return numpy.inf


# noinspection PyAbstractClass
class BackpropagationFitness(evo.Fitness):
    """A fitness for symbolic regression that operates with solutions made of
    trees that support backpropagation.
    """

    LOG = logging.getLogger(__name__ + '.BackpropagationFitness')

    def __init__(self, error_fitness, handled_errors, cost_derivative,
                 updater: evo.sr.backpropagation.WeightsUpdater, steps=10,
                 min_steps=0, fit: bool=False,
                 synchronize_lincomb_vars: bool=False,
                 stats: evo.utils.stats.Stats=None, store_bsfs: bool=True,
                 backpropagate_only: bool=False):
        """
        The ``var_mapping`` argument is responsible for mapping the input
        variables to variable names of a tree. It is supposed to be a dict with
        keys being integers counting from 0 and values being variable names in
        the tree. The keys are supposed to correspond to the column indices to
        ``train_inputs``.

        :param error_fitness: the fitness value that is assigned to the
            individuals which produce :class:`ZeroDivisionError`\ ,
            :class:`FloatingPointError` or any other error specified in
            *handled_errors* argument
        :param handled_errors: additional errors (exceptions) that should be
            caught when applying a gene to the data
        :param steps: number of steps the optimisation algorithm will do prior
            to each error evaluation

            There are several forms of this setting:

                * integer value - this just sets the number of steps that
                  will be performed
                * ``('depth', max_steps)`` - the number of steps that will be
                  performed is determined as the ``max_steps`` minus the
                  (maximum) depth of the tree
                * ``('nodes', max_steps)`` - the number of steps that will be
                  performed is determined as the ``max_steps`` minus the
                  number of nodes in the tree
        :param min_steps: minimum number of steps the optimisation algorithm
            will *always* do prior to each error evaluation (regardles what
            the ``steps`` argument is set to); default is 0
        :param fit: if ``True`` the output of the genomes will be
            additionally transformed using the :meth:`.fit_outputs` method.
        :param synchronize_lincomb_vars: if ``True`` the leaves of type
            :class:`evo.sr.backpropagation.LincombVariable` are
            *syncrhonized*\ , meaning that after the backpropagation the
            partial derivatives are summed up and set to those with the same
            index, effectively making a single affine transformation from all
            these leaf nodes
        """
        super().__init__(store_bsfs)
        self.error_fitness = error_fitness
        self.errors = tuple([ZeroDivisionError, FloatingPointError] +
                            handled_errors)
        self.cost_derivative = cost_derivative
        self.updater = updater
        if isinstance(steps, int):
            self.steps = steps
            self.get_max_steps = self.steps_number
        elif steps[0] == 'depth':
            self.steps = steps[1]
            self.get_max_steps = self.steps_depth
        elif steps[0] == 'nodes':
            self.steps = steps[1]
            self.get_max_steps = self.steps_nodes
        self.min_steps = min_steps
        self.fit = fit
        self.synchronize_lincomb_vars = synchronize_lincomb_vars
        self.backpropagate_only = backpropagate_only

        self.stats = stats

    def get_train_inputs(self):
        raise NotImplementedError()

    def get_train_input_cases(self) -> int:
        raise NotImplementedError()

    def get_train_output(self):
        raise NotImplementedError()

    def get_args(self):
        return self.get_train_inputs()

    def steps_number(self, individual: FittedForestIndividual):
        ret = self.steps
        return ret

    def steps_depth(self, individual: FittedForestIndividual):
        ret = self.steps - sum(g.get_subtree_depth() for g
                               in individual.genotype)
        return ret

    def steps_nodes(self, individual: FittedForestIndividual):
        ret = self.steps - sum(g.get_subtree_size() for g
                               in individual.genotype)
        return ret

    def evaluate_individual(self, individual: FittedForestIndividual,
                            context=None):
        BackpropagationFitness.LOG.debug(
            'Evaluating individual %s in context %s', individual.__str__(),
            str(context))

        try:
            fitness, otf, otf_d = self._eval_individual(
                individual, self.synchronize_lincomb_vars, context)
            prev_fitness = fitness
            BackpropagationFitness.LOG.debug('Optimising inner parameters...')
            BackpropagationFitness.LOG.debug('Initial fitness: %f', fitness)
            if self.backpropagate_only:
                self.backpropagate_bases(individual, otf, otf_d)
                steps = 0
            else:
                steps = self.get_max_steps(individual)
                if steps < self.min_steps:
                    steps = self.min_steps
            for i in range(steps):
                self.backpropagate_bases(individual, otf, otf_d)
                updated = self.update_bases(fitness, individual, prev_fitness)

                if not updated:
                    BackpropagationFitness.LOG.debug(
                        'No update occurred, stopping inner learning.')
                    break

                prev_fitness = fitness
                fitness, otf, otf_d = self._eval_individual(individual, False)
                if not self.has_improved(prev_fitness, fitness):
                    BackpropagationFitness.LOG.debug('Improvement below '
                                                     'threshold, stopping '
                                                     'inner learning.')
                    break
                BackpropagationFitness.LOG.debug(
                    'Step %d fitness: %f Full model: %s', i, fitness,
                    [g.infix() for g in individual.genotype])
            individual.set_fitness(fitness)
        except self.errors as e:
            BackpropagationFitness.LOG.debug(
                'Exception occurred during evaluation, assigning fitness %f',
                self.error_fitness, exc_info=True)
            fitness = self.error_fitness
            individual.set_fitness(fitness)
        except evo.UnevaluableError as e:
            pass
        return individual.get_fitness()

    def _eval_individual(self, individual, pick_lincombs: bool, context=None):
        if pick_lincombs:
            do = False
            lcs = list(itertools.chain.from_iterable(
                [root.get_nodes_dfs(predicate=lambda n: isinstance(
                    n, evo.sr.backpropagation.LincombVariable))
                 for root in individual.genotype]))
            key = operator.attrgetter('index')
            lcs.sort(key=key)
            for i, g in itertools.groupby(lcs, key=key):
                g = list(g)
                if len(g) <= 1:
                    continue
                values = []
                for v in g:
                    values.append((numpy.copy(v.bias), numpy.copy(v.weights)))
                b = []
                w = []
                subkey = lambda x: (list(x[0]), list(x[1]))
                values.sort(key=subkey)
                for j, g2 in itertools.groupby(values,
                                               key=subkey):
                    g2 = list(g2)
                    b.append(g2[0][0])
                    w.append(g2[0][1])
                if len(b) > 1 or len(w) > 1:
                    do = True
                else:
                    continue

                b.append((b[0] + b[1]) / 2)
                w.append((w[0] + w[1]) / 2)
                for j, v in enumerate(g):
                    v.data['b'] = b
                    v.data['w'] = w
            f_best = None
            i_best = None
            otf_best = None
            otf_d_best = None
            if do:
                for i in range(3):
                    for n in lcs:
                        if 'b' in n.data:
                            n.bias = n.data['b'][i]
                            n.notify_change()
                        if 'w' in n.data:
                            n.weights = n.data['w'][i]
                            n.notify_change()
                    f, otf, otf_d = self._eval_individual(individual, False)
                    if i_best is None or self.fitness_cmp(f, f_best) < 0:
                        f_best = f
                        i_best = i
                        otf_best = otf
                        otf_d_best = otf_d
                for n in lcs:
                    if 'b' in n.data:
                        n.bias = n.data['b'][i_best]
                        n.notify_change()
                        del n.data['b']
                    if 'w' in n.data:
                        n.weights = n.data['w'][i_best]
                        n.notify_change()
                        del n.data['w']
                return f_best, otf_best, otf_d_best

        yhats = self.get_eval(individual, self.get_args())
        if self.stats is not None:
            iteration = -1
            runtime = -1
            # noinspection PyBroadException
            try:
                iteration = context.iterations
                runtime = context.get_runtime()
            except:
                pass
            report_data = [runtime, iteration,
                           [str(g) for g in individual.genotype],
                           full_model_str(individual)]
            self.stats.report_data(report_data)

        BackpropagationFitness.LOG.debug('Checking output...')
        self._check_output(yhats, individual)

        otf = otf_d = lambda _: None
        if self.fit:
            BackpropagationFitness.LOG.debug('Performing output fitting...')
            self.fit_outputs(individual, yhats)
            otf, otf_d = self.get_output_transformation(individual, yhats)
        fitness = self.get_error(yhats, individual)
        return fitness, otf, otf_d

    def backpropagate_bases(self, individual, transform, tansform_derivative):
        for n in range(individual.genes_num):
            try:
                base = individual.genotype[n]
                base.backpropagate(
                    args=self.get_args(),
                    datapts_no=self.get_train_input_cases(),
                    cost_derivative=self.cost_derivative,
                    true_output=self.get_train_output(),
                    output_transform=transform(n),
                    output_transform_derivative=tansform_derivative(n)
                )
            except AttributeError:
                continue
        if self.synchronize_lincomb_vars:
            self.synchronize_lincombs(individual.genotype)

    def synchronize_lincombs(self, roots: list):
        lcs = list(itertools.chain.from_iterable(
            [root.get_nodes_dfs(predicate=lambda n: isinstance(
                n, evo.sr.backpropagation.LincombVariable))
             for root in roots]))
        key = operator.attrgetter('index')
        lcs.sort(key=key)
        for i, g in itertools.groupby(lcs, key=key):
            g = list(g)
            if len(g) <= 1:
                continue
            b = 0.0
            w = 0.0
            for v in g:
                b = b + v.data['d_bias']
                w = w + v.data['d_weights']
            for v in g:
                v.data['d_bias'] = numpy.copy(b)
                v.data['d_weights'] = numpy.copy(w)

    def update_bases(self, fitness, individual, prev_fitness):
        updated = False
        for n in range(individual.genes_num):
            gene_updated = self.update_base(fitness, individual.genotype[n],
                                            prev_fitness)
            updated = updated or gene_updated
        return updated

    def update_base(self, fitness, base, prev_fitness):
        return self.updater.update(base, fitness, prev_fitness)

    def get_eval(self, individual: FittedForestIndividual, args):
        if individual.genes_num == 1:
            return individual.genotype[0].eval(args=args)
        yhats = [g.eval(args=args) for g in individual.genotype]
        yhats2 = evo.utils.column_stack(*yhats)
        return yhats2

    def _check_output(self, yhat, individual: FittedForestIndividual):
        if numpy.any(numpy.logical_or(numpy.isinf(yhat),
                                      numpy.isnan(yhat))):
            BackpropagationFitness.LOG.debug('NaN or inf in output, assigning '
                                             'fitness: %f', self.error_fitness)
            individual.set_fitness(self.error_fitness)
            raise evo.UnevaluableError()

    def get_error(self, outputs, individual: FittedForestIndividual):
        """Computes the error of the individual.

        :param outputs: a vector of outputs of the individual
        :param individual: the individual which is analyzed
        :return: the resulting error
        """
        raise NotImplementedError()

    def compare(self, i1, i2, context=None):
        f1 = i1.get_fitness()
        f2 = i2.get_fitness()
        if f1 is None:
            self.evaluate(i1, context)
            f1 = i1.get_fitness()
        if f2 is None:
            self.evaluate(i2, context)
            f2 = i2.get_fitness()
        return self.fitness_cmp(f1, f2)

    def fitness_cmp(self, f1, f2):
        raise NotImplementedError()

    def fit_outputs(self, individual: FittedForestIndividual, outputs):
        raise NotImplementedError()

    def get_output_transformation(self, individual: FittedForestIndividual,
                                  yhats):
        raise NotImplementedError()

    @staticmethod
    def has_improved(prev_fitness, fitness):
        """Returns ``true`` if the fitness is considered to be improved
        compared to the previous fitness.
        """
        return abs(prev_fitness - fitness) > 1e-10


class RegressionFitness(BackpropagationFitness):
    """This is a class that uses backpropagation to fit to static target values.
    """

    LOG = logging.getLogger(__name__ + '.RegressionFitness')

    def __init__(self, handled_errors, train_inputs, train_output,
                 updater: evo.sr.backpropagation.WeightsUpdater, steps=10,
                 min_steps=0, fit: bool=False,
                 synchronize_lincomb_vars: bool=False,
                 stats: evo.utils.stats.Stats=None,
                 fitness_measure: ErrorMeasure=ErrorMeasure.R2,
                 backpropagate_only: bool=False):
        """
        :param train_inputs: feature variables' values: an N x M matrix where N
            is the number of datapoints (the same N as in ``target`` argument)
            and M is the number of feature variables
        :param train_output: target values of the datapoints: an N x 1 matrix
            where N is the number of datapoints
        :param fitness_measure: specifies which error measure is going to be
            used as fitness, including the correct comparison for determining
            the better of two individuals
        """
        super().__init__(fitness_measure.worst, handled_errors,
                         lambda yhat, y: yhat - y, updater, steps, min_steps,
                         fit, synchronize_lincomb_vars, stats,
                         backpropagate_only=backpropagate_only)
        self.train_inputs = train_inputs
        self.train_output = numpy.array(train_output, copy=False)
        self.ssw = numpy.sum(
            (self.train_output - self.train_output.mean()) ** 2)
        self.fitness_measure = fitness_measure

    def get_train_inputs(self):
        return self.train_inputs

    def get_train_output(self):
        return self.train_output

    def get_train_input_cases(self):
        return self.train_inputs.shape[0]

    def get_args(self):
        return self.train_inputs

    def get_error(self, outputs, individual: FittedForestIndividual):
        e = self.get_errors(outputs, individual)
        ae = numpy.abs(e)
        sse = e.dot(e)
        r2 = 1 - sse / self.ssw
        mse = sse / numpy.alen(e)
        mae = numpy.sum(ae) / numpy.alen(e)
        worst_case_ae = ae.max()
        individual.set_data('R2', r2)
        individual.set_data('MSE', mse)
        individual.set_data('MAE', mae)
        individual.set_data('WORST_CASE_AE', worst_case_ae)
        if self.fitness_measure is ErrorMeasure.R2:
            return r2
        if self.fitness_measure is ErrorMeasure.MSE:
            return mse
        if self.fitness_measure is ErrorMeasure.MAE:
            return mae
        if self.fitness_measure is ErrorMeasure.WORST_CASE_AE:
            return worst_case_ae
        raise ValueError('Invalid value of fitness_measure.')

    def get_output(self, outputs, individual: FittedForestIndividual):
        intercept = individual.intercept
        coefficients = individual.coefficients
        output = None
        if coefficients is not None:
            if not isinstance(outputs, numpy.ndarray) or outputs.ndim == 1:
                output = outputs * coefficients
            else:
                output = outputs.dot(coefficients)
        if intercept is not None:
            output = output + intercept
        return output

    def get_errors(self, outputs, individual: FittedForestIndividual):
        output = self.get_output(outputs, individual)
        return self.get_train_output() - output

    def fit_outputs(self, individual: FittedForestIndividual, outputs):
        target = self.get_train_output()
        if not isinstance(outputs, numpy.ndarray):
            base = numpy.empty((target.shape[0], 2))
        elif outputs.ndim == 1:
            base = numpy.empty((target.shape[0], 2))
            outputs = outputs[:, numpy.newaxis]
        else:
            base = numpy.empty((target.shape[0], outputs.shape[1] + 1))
        base[:, 0] = 1
        base[:, 1:] = outputs
        w = numpy.linalg.lstsq(base, target)[0]
        individual.intercept = w[0]
        individual.coefficients = w[1:]

    def get_output_transformation(self, individual: FittedForestIndividual,
                                  yhats):
        def otf(n):
            def f(y):
                i = individual.intercept
                c = individual.coefficients
                if c.size == 1:
                    return i + c[0] * y
                mask = numpy.ones(len(c), dtype=bool)
                mask[n] = False
                cn = c[n]
                co = c[mask]
                return i + cn * y + yhats[:, mask].dot(co)

            return f

        def otf_d(n):
            return lambda _: individual.coefficients[n]

        return otf, otf_d

    def fitness_cmp(self, f1, f2):
        if self.fitness_measure is ErrorMeasure.R2:
            if f1 > f2:
                return -1
            if f1 < f2:
                return 1
        else:
            if f1 < f2:
                return -1
            if f1 > f2:
                return 1
        return 0


# experimental, not maintained for now
# noinspection PyTypeChecker,PyArgumentList,PyAttributeOutsideInit
class RegressionFitnessRst(RegressionFitness):

    def __init__(self, handled_errors, train_inputs, train_output,
                 updater: evo.sr.backpropagation.WeightsUpdater, steps=10,
                 min_steps=0, fit: bool = False):
        super().__init__(handled_errors, train_inputs, train_output, updater,
                         steps, min_steps, fit)
        self.full_ssw = self.ssw
        self.full_inputs = self.train_inputs
        self.full_output = self.train_output

    def resample_subset(self, generator):
        self.subset_frac = 0.8
        n = self.full_inputs.shape[0]
        perm = generator.sample(range(n), max(1, int(n * self.subset_frac)))
        self.train_inputs = self.full_inputs[perm, :]
        self.train_output = self.full_output[perm]
        self.ssw = numpy.sum(
            (self.train_output - self.train_output.mean()) ** 2)

    def evaluate_individual(self, individual: evo.gp.support.ForestIndividual,
                            context=None):
        for g in individual.genotype:
            g.clear_cache()
        super().evaluate(individual)
        subset_ssw = self.ssw
        self.ssw = self.full_ssw
        subset_tr_in = self.train_inputs
        self.train_inputs = self.full_inputs
        subset_tr_out = self.train_output
        self.train_output = self.full_output

        try:
            for g in individual.genotype:
                g.clear_cache()
            yhats = self.get_eval(individual)
            check = self._check_output(yhats, individual)
            if check is not None:
                return check
            if self.fit:
                self.fit_outputs(individual, yhats)
            fitness = self.get_error(yhats, individual)
        except self.errors:
            BackpropagationFitness.LOG.debug(
                'Exception occurred during evaluation, assigning fitness %f',
                self.error_fitness, exc_info=True)
            fitness = self.error_fitness
        individual.set_fitness(fitness)

        self.ssw = subset_ssw
        self.train_inputs = subset_tr_in
        self.train_output = subset_tr_out

        return fitness


def full_model_str(individual: FittedForestIndividual,
                   **kwargs) -> str:
    nf = kwargs.get('num_format', '.3f')
    ic = individual.intercept
    co = individual.coefficients
    if nf == 'repr':
        strs = ['{}'.format(repr(ic))]
        for c, g in zip(co, individual.genotype):
            strs.append('{} * {}'.format(repr(c), g.infix(**kwargs)))
    else:
        strs = [('{:' + nf + '}').format(ic)]
        for c, g in zip(co, individual.genotype):
            strs.append(('{:' + nf + '} * {}').format(c, g.infix(**kwargs)))
    return ' + '.join(strs)


class GlobalLincombsGp(evo.gp.Gp):

    LOG = logging.getLogger(__name__ + '.GlobalLincombsGp')

    def __init__(self, fitness: evo.Fitness,
                 pop_strategy: evo.PopulationStrategy,
                 selection_strategy: evo.SelectionStrategy,
                 reproduction_strategy: evo.ReproductionStrategy,
                 population_initializer: evo.PopulationInitializer, functions,
                 terminals, stop, coeff_mut_prob, coeff_mut_sigma, global_lcs,
                 update_steps, **kwargs):
        super().__init__(fitness, pop_strategy, selection_strategy,
                         reproduction_strategy, population_initializer,
                         functions, terminals, stop, **kwargs)
        self.coeff_mut_prob = coeff_mut_prob
        self.cm = CoefficientsMutation(coeff_mut_sigma, self.generator)
        self.global_lcs = global_lcs
        self.update_steps = update_steps

    def _iteration(self):
        GlobalLincombsGp.LOG.debug('Starting iteration %d', self.iterations)
        self.try_stop()
        if self.callback is not None:
            self.callback(self, evo.gp.Gp.CallbackSituation.iteration_start)

        self._eval_all()
        GlobalLincombsGp.LOG.debug('Before global update: BSF %s | %s | %s',
                                   self.fitness.get_bsf().get_fitness(),
                                   str(self.fitness.get_bsf()),
                                   self.fitness.get_bsf().get_data())
        for i in range(self.update_steps):
            self._synchronize()
            self._update(0, 0)
            self._eval_all()
            GlobalLincombsGp.LOG.debug('After global update step %d / %d: BSF '
                                       '%s | %s | %s', i + 1, self.update_steps,
                                       self.fitness.get_bsf().get_fitness(),
                                       str(self.fitness.get_bsf()),
                                       self.fitness.get_bsf().get_data())

        elites = self.top_individuals(self.pop_strategy.get_elites_number())

        GlobalLincombsGp.LOG.debug('Processing selection.')
        offspring = []
        while len(offspring) < self.pop_strategy.get_offspring_number():
            self.try_stop()
            self.reproduction_strategy.reproduce(self.selection_strategy,
                                                 self.pop_strategy,
                                                 self,
                                                 self.population,
                                                 offspring)
        self._post_synchronize(offspring)
        self.population = self.pop_strategy.combine_populations(
            self.population, offspring, elites)
        self._mutate()
        GlobalLincombsGp.LOG.info('Iteration %d / %.1f s. BSF %s | '
                                  '%s | %s',
                                  self.iterations, self.get_runtime(),
                                  self.fitness.get_bsf().get_fitness(),
                                  str(self.fitness.get_bsf()),
                                  self.fitness.get_bsf().get_data())
        if self.callback is not None:
            self.callback(self, evo.gp.Gp.CallbackSituation.iteration_end)
        self.iterations += 1

    def eval_individual(self, i):
        self.try_stop()
        return self.fitness.evaluate(i, context=self)

    def _eval_all(self):
        for i in self.population:
            self.eval_individual(i)

    def _synchronize(self):
        all_nodes = []
        for i in self.population:
            for g in i.genotype:
                all_nodes.extend(g.get_nodes_dfs(
                    predicate=self._node_filter))
        d_bias = {}
        d_weights = {}
        prev_d_bias = {}
        prev_d_weights = {}
        for n in all_nodes:
            i = n.index
            if 'd_bias' in n.data:
                if i not in d_bias:
                    d_bias[i] = n.data['d_bias']
                else:
                    d_bias[i] = d_bias[i] + n.data['d_bias']
            if 'd_weights' in n.data:
                if i not in d_weights:
                    d_weights[i] = n.data['d_weights']
                else:
                    d_weights[i] = d_weights[i] + n.data['d_weights']
            if 'prev_d_bias' in n.data:
                prev_d_bias[i] = n.data['prev_d_bias']
            if 'prev_d_weights' in n.data:
                prev_d_weights[i] = n.data['prev_d_weights']

        for n in all_nodes:
            i = n.index
            if i in d_bias:
                n.data['d_bias'] = numpy.copy(d_bias[i])
            if i in d_weights:
                n.data['d_weights'] = numpy.copy(d_weights[i])
            if i in prev_d_bias:
                n.data['prev_d_bias'] = numpy.copy(prev_d_bias[i])
            if i in prev_d_weights:
                n.data['prev_d_weights'] = numpy.copy(prev_d_weights[i])

        for n in self.global_lcs:
            i = n.index
            if i in d_bias:
                n.data['d_bias'] = d_bias[i]
            if i in d_weights:
                n.data['d_weights'] = d_weights[i]
            if i in prev_d_bias:
                n.data['prev_d_bias'] = prev_d_bias[i]
            if i in prev_d_weights:
                n.data['prev_d_weights'] = prev_d_weights[i]

        for n in all_nodes:
            i = n.index
            global_n = self.global_lcs[i]
            if 'delta_bias' in global_n.data:
                n.data['delta_bias'] = numpy.copy(global_n.data['delta_bias'])
            if 'delta_weights' in global_n.data:
                n.data['delta_weights'] = numpy.copy(
                    global_n.data['delta_weights'])

    def _post_synchronize(self, offspring):
        all_nodes = []
        for i in offspring:
            for g in i.genotype:
                all_nodes.extend(g.get_nodes_dfs(
                    predicate=self._node_filter))
        for n in all_nodes:
            i = n.index
            global_n = self.global_lcs[i]
            n.bias = numpy.copy(global_n.bias)
            n.weights = numpy.copy(global_n.weights)

    @staticmethod
    def _node_filter(n):
        return isinstance(n, evo.sr.backpropagation.LincombVariable)

    def _update(self, prev_bsf_fitness, cur_bsf_fitness):
        for n in self.global_lcs:
            self.fitness.update_base(cur_bsf_fitness, n, prev_bsf_fitness)
        for i in self.population:
            self.fitness.update_bases(cur_bsf_fitness, i, prev_bsf_fitness)

    def _mutate(self):
        if self.generator.random() >= self.coeff_mut_prob:
            return
        for lc in self.global_lcs:
            self.cm.mutate_node(lc, self.cm.sigma)
        for i in self.population:
            for g in i.genotype:
                for n in g.get_nodes_dfs(predicate=self._node_filter):
                    n.bias[:] = self.global_lcs[n.index].bias[:]
                    n.weights[:] = self.global_lcs[n.index].weights[:]
                    n.notify_change()
