# -*- coding: utf8 -*-
"""TODO
"""

import functools
import logging

import numpy

import evo
import evo.gp
import evo.gp.support
import evo.sr
import evo.sr.backpropagation


# noinspection PyAbstractClass
class BackpropagationFitness(evo.Fitness):
    """A fitness for symbolic regression that operates with solutions made of
    trees that support backpropagation.
    """

    LOG = logging.getLogger(__name__ + '.BackpropagationFitness')

    def __init__(self, error_fitness, handled_errors, cost_derivative,
                 updater: evo.sr.backpropagation.WeightsUpdater, steps=10,
                 min_steps=0, fit: bool=False):
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
        """
        super().__init__()
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

        self.bsf = None

    def get_train_inputs(self):
        raise NotImplementedError()

    def get_train_input_cases(self) -> int:
        raise NotImplementedError()

    def get_train_output(self):
        raise NotImplementedError()

    def get_args(self):
        return self.get_train_inputs()

    def steps_number(self, individual):
        ret = self.steps
        return ret

    def steps_depth(self, individual):
        ret = self.steps - sum(g.get_subtree_depth() for g
                               in individual.genotype)
        return ret

    def steps_nodes(self, individual):
        ret = self.steps - sum(g.get_subtree_size() for g
                               in individual.genotype)
        return ret

    def evaluate(self, individual: evo.gp.support.ForestIndividual):
        BackpropagationFitness.LOG.debug('Evaluating individual %s',
                                         individual.__str__())

        otf = lambda _: None
        otf_d = lambda _: None
        try:
            yhats = self.get_eval(individual)
            BackpropagationFitness.LOG.debug('Checking output...')
            check = self._check_output(yhats, individual)
            if check is not None:
                return check
            if self.fit:
                BackpropagationFitness.LOG.debug('Performing output fitting...')
                self.fit_outputs(individual, yhats)
                otf, otf_d = self.get_output_transformation(individual, yhats)
            fitness = prev_fitness = self.get_error(yhats, individual)
            BackpropagationFitness.LOG.debug('Optimising inner parameters...')
            BackpropagationFitness.LOG.debug('Initial fitness: %f', fitness)
            for i in range(self.min_steps, max(self.get_max_steps(individual),
                                               self.min_steps + 1)):
                updated = False
                for n in range(individual.genes_num):
                    try:
                        individual.genotype[n].backpropagate(
                            args=self.get_args(),
                            datapts_no=self.get_train_input_cases(),
                            cost_derivative=self.cost_derivative,
                            true_output=self.get_train_output(),
                            output_transform=otf(n),
                            output_transform_derivative=otf_d(n)
                        )
                    except AttributeError:
                        continue
                    gene_updated = self.updater.update(individual.genotype[n],
                                                       fitness, prev_fitness)
                    updated = updated or gene_updated

                if not updated:
                    BackpropagationFitness.LOG.debug(
                        'No update occurred, stopping inner learning.')
                    break
                yhats = self.get_eval(individual)
                check = self._check_output(yhats, individual)
                if check is not None:
                    return check
                if self.fit:
                    self.fit_outputs(individual, yhats)
                    otf, otf_d = self.get_output_transformation(individual,
                                                                yhats)
                prev_fitness = fitness
                fitness = self.get_error(yhats, individual)
                if not self.has_improved(prev_fitness, fitness):
                    BackpropagationFitness.LOG.debug('Improvement below '
                                                     'threshold, stopping '
                                                     'inner learning.')
                    break
                BackpropagationFitness.LOG.debug(
                    'Step %d fitness: %f Full model: %s', i, fitness,
                    [g.infix() for g in individual.genotype])
        except self.errors as e:
            BackpropagationFitness.LOG.debug(
                'Exception occurred during evaluation, assigning fitness %f',
                self.error_fitness, exc_info=True)
            fitness = self.error_fitness
        individual.set_fitness(fitness)
        if self.bsf is None or self.compare(individual, self.bsf) < 0:
            self.bsf = individual.copy()
        return fitness

    def get_eval(self, individual):
        args = self.get_args()
        if individual.genes_num == 1:
            return individual.genotype[0].eval(args=args)
        yhats = [g.eval(args=args) for g in individual.genotype]
        if len(yhats) == 1:
            yhats = yhats[0][:, numpy.newaxis]
        yhats = numpy.column_stack(numpy.broadcast(*yhats)).T
        return yhats

    def _check_output(self, yhat, individual):
        if numpy.any(numpy.logical_or(numpy.isinf(yhat),
                                      numpy.isnan(yhat))):
            BackpropagationFitness.LOG.debug('NaN or inf in output, assigning '
                                             'fitness: %f', self.error_fitness)
            individual.set_fitness(self.error_fitness)
            return self.error_fitness
        return None

    def get_error(self, outputs, individual):
        """Computes the error of the individual.

        :param outputs: a vector of outputs of the individual
        :param individual: the individual which is analyzed
        :return: the resulting error
        """
        raise NotImplementedError()

    def sort(self, population, reverse=False, *args):
        population.sort(key=functools.cmp_to_key(self.compare))
        return True

    def compare(self, i1, i2, *args):
        f1 = i1.get_fitness()
        f2 = i2.get_fitness()
        if f1 is None:
            self.evaluate(i1)
            f1 = i1.get_fitness()
        if f2 is None:
            self.evaluate(i2)
            f2 = i2.get_fitness()
        return self.fitness_cmp(f1, f2)

    def fitness_cmp(self, f1, f2):
        raise NotImplementedError()

    def get_bsf(self) -> evo.Individual:
        return self.bsf

    def fit_outputs(self, individual, outputs):
        raise NotImplementedError()

    def get_output_transformation(self, individual, yhats):
        raise NotImplementedError()

    def has_improved(self, prev_fitness, fitness):
        """Returns ``true`` if the fitness is considered to be improved
        compared to the previous fitness.
        """
        return abs(prev_fitness - fitness) > 1e-10


class RegressionFitness(BackpropagationFitness):
    """This is a class that uses backpropagation to fit to static target values.
    """

    def __init__(self, handled_errors, train_inputs, train_output,
                 updater: evo.sr.backpropagation.WeightsUpdater, steps=10,
                 min_steps=0, fit: bool=False):
        """
        :param train_inputs: feature variables' values: an N x M matrix where N
            is the number of datapoints (the same N as in ``target`` argument)
            and M is the number of feature variables
        :param train_output: target values of the datapoints: an N x 1 matrix
            where N is the number of datapoints
        """
        super().__init__(-numpy.inf, handled_errors, lambda yhat, y: yhat - y,
                         updater, steps, min_steps, fit)
        self.train_inputs = train_inputs
        self.train_output = numpy.array(train_output, copy=False)
        self.ssw = numpy.sum(
            (self.train_output - self.train_output.mean()) ** 2)

    def get_train_inputs(self):
        return self.train_inputs

    def get_train_output(self):
        return self.train_output

    def get_train_input_cases(self):
        return self.train_inputs.shape[0]

    def get_args(self):
        return self.train_inputs

    def get_error(self, outputs, individual):
        e = self.get_errors(outputs, individual)
        sse = e.dot(e)
        r2 = 1 - sse / self.ssw
        mse = sse / numpy.alen(e)
        individual.set_data('R2', r2)
        individual.set_data('MSE', mse)
        return r2

    def get_errors(self, outputs, individual):
        intercept = individual.get_data('intercept')
        coefficients = individual.get_data('coefficients')
        if coefficients is not None:
            if not isinstance(outputs, numpy.ndarray) or outputs.ndim == 1:
                outputs = outputs * coefficients
            else:
                outputs = outputs.dot(coefficients)
        if intercept is not None:
            outputs = outputs + intercept
        return self.get_train_output() - outputs

    def fit_outputs(self, individual, outputs):
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
        individual.set_data('intercept', w[0])
        individual.set_data('coefficients', w[1:])

    def get_output_transformation(self, individual, yhats):
        def otf(n):
            def f(y):
                i = individual.get_data('intercept')
                c = individual.get_data('coefficients')
                if c.size == 1:
                    return i + c[0] * y
                mask = numpy.ones(len(c), dtype=bool)
                mask[n] = False
                cn = c[n]
                co = c[mask]
                return i + cn * y + yhats[:, mask].dot(co)

            return f

        def otf_d(n):
            return lambda _: individual.get_data('coefficients')[n]

        return otf, otf_d

    def fitness_cmp(self, f1, f2):
        if f1 > f2:
            return -1
        if f1 < f2:
            return 1
        return 0


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

    def evaluate(self, individual: evo.gp.support.ForestIndividual):
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
        except self.errors as e:
            BackpropagationFitness.LOG.debug(
                'Exception occurred during evaluation, assigning fitness %f',
                self.error_fitness, exc_info=True)
            fitness = self.error_fitness
        individual.set_fitness(fitness)
        if self.bsf is None or self.compare(individual, self.bsf) < 0:
            self.bsf = individual.copy()

        self.ssw = subset_ssw
        self.train_inputs = subset_tr_in
        self.train_output = subset_tr_out

        return fitness


class RegressionFitnessMaxError(RegressionFitness):
    class _Updater(object):
        def __init__(self, upd):
            self.updater = upd

        def update(self, root, error, prev_error):
            return self.updater.update(root, error[1], prev_error[1])

    def __init__(self, handled_errors, train_inputs, train_output,
                 updater: evo.sr.backpropagation.WeightsUpdater, steps=10,
                 min_steps=0, fit: bool = False):
        super().__init__(handled_errors, train_inputs, train_output, updater,
                         steps, min_steps, fit)

        # noinspection PyProtectedMember
        self.updater = self.__class__._Updater(self.updater)
        self.error_fitness = (numpy.inf, self.error_fitness)

    def get_error(self, outputs, individual):
        e = self.get_errors(outputs, individual)
        sse = e.dot(e)
        r2 = 1 - sse / self.ssw
        mse = sse / numpy.alen(e)
        individual.set_data('R2', r2)
        individual.set_data('MSE', mse)
        max_e = numpy.abs(e).max()
        return max_e, r2

    def fitness_cmp(self, f1, f2):
        if f1[0] < f2[0]:
            return -1
        if f1[0] > f2[0]:
            return 1
        return super().fitness_cmp(f1[1], f2[1])

    def has_improved(self, prev_fitness, fitness):
        return super().has_improved(prev_fitness[1], fitness[1])


def full_model_str(individual: evo.gp.support.ForestIndividual,
                   **kwargs) -> str:
    nf = kwargs.get('num_format', '.3f')
    ic = individual.get_data('intercept')
    co = individual.get_data('coefficients')
    if nf == 'repr':
        strs = ['{}'.format(repr(ic))]
        for c, g in zip(co, individual.genotype):
            strs.append('{} * {}'.format(repr(c), g.infix(**kwargs)))
    else:
        strs = [('{:' + nf + '}').format(ic)]
        for c, g in zip(co, individual.genotype):
            strs.append(('{:' + nf + '} * {}').format(c, g.infix(**kwargs)))
    return ' + '.join(strs)
