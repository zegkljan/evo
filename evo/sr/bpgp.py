# -*- coding: utf8 -*-
"""TODO
"""

import logging
import numpy
import functools

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

    def __init__(self, error_fitness, handled_errors, var_mapping: dict,
                 updater: evo.sr.backpropagation.WeightsUpdater,
                 steps: int=10):
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
        :param var_mapping: mapping from :class:`int` to :class:`str` that maps
            a number of an input variable to variable name (the number of
            feature is the number of corresponding column in the
            ``train_inputs`` argument)
        :param steps: number of steps the optimisation algorithm will do prior
            to each error evaluation
        """
        super().__init__()
        self.error_fitness = error_fitness
        self.var_mapping = var_mapping
        self.errors = tuple([ZeroDivisionError, FloatingPointError] +
                            handled_errors)
        self.updater = updater
        self.steps = steps

        self.cost_derivative = lambda yhat, y: yhat - y
        self.bsf = None

    def get_train_inputs(self):
        raise NotImplementedError()

    def get_train_input_cases(self) -> int:
        raise NotImplementedError()

    def get_train_output(self):
        raise NotImplementedError()

    def get_args(self):
        inputs = self.get_train_inputs()
        return evo.sr.prepare_args(inputs, self.var_mapping)

    def evaluate(self, individual: evo.gp.support.TreeIndividual):
        BackpropagationFitness.LOG.debug('Evaluating individual %s',
                                         individual.__str__())

        BackpropagationFitness.LOG.debug('Optimising inner parameters.')
        try:
            yhat = individual.genotype.eval(args=self.get_args())
            check = self._check_output(yhat, individual)
            if check is not None:
                return check
            fitness = prev_error = self.get_error(yhat, individual)
            BackpropagationFitness.LOG.debug('Initial error: %f', fitness)
            for i in range(self.steps):
                evo.sr.backpropagation.backpropagate(
                    individual.genotype, self.cost_derivative,
                    self.get_train_output(), self.get_args(),
                    self.get_train_input_cases())
                updated = self.updater.update(individual.genotype,
                                              fitness, prev_error)

                if not updated:
                    BackpropagationFitness.LOG.debug(
                        'Stopping rprop because no update occurred.')
                    break
                yhat = individual.genotype.eval(args=self.get_args())
                check = self._check_output(yhat, individual)
                if check is not None:
                    return check
                prev_error = fitness
                fitness = self.get_error(yhat, individual)
                BackpropagationFitness.LOG.debug(
                    'Step %d error: %f Full model: %s', i, fitness,
                    individual.genotype.full_infix())
        except self.errors as e:
            BackpropagationFitness.LOG.debug(
                'Exception occurred during evaluation, assigning fitness %f',
                self.error_fitness, exc_info=True)
            fitness = self.error_fitness
        individual.set_fitness(fitness)
        if self.bsf is None or self.bsf.get_fitness() > fitness:
            self.bsf = individual.copy()
        return fitness

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
        return f1 < f2

    def get_bsf(self) -> evo.Individual:
        return self.bsf


class RegressionFitness(BackpropagationFitness):
    """This is a class that uses backpropagation to fit to static target values.
    """
    def __init__(self, error_fitness, handled_errors, train_inputs,
                 train_output, var_mapping: dict,
                 updater: evo.sr.backpropagation.WeightsUpdater,
                 steps: int = 10):
        """
        :param train_inputs: feature variables' values: an N x M matrix where N
            is the number of datapoints (the same N as in ``target`` argument)
            and M is the number of feature variables
        :param train_output: target values of the datapoints: an N x 1 matrix
            where N is the number of datapoints
        """
        super().__init__(error_fitness, handled_errors, var_mapping, updater,
                         steps)
        self.train_inputs = train_inputs
        self.train_output = numpy.array(train_output, copy=False)
        self.args = evo.sr.prepare_args(train_inputs, var_mapping)

    def get_train_inputs(self):
        return self.train_inputs

    def get_train_output(self):
        return self.train_output

    def get_train_input_cases(self):
        return self.train_inputs.shape[0]

    def get_args(self):
        return self.args

    def get_error(self, outputs, individual):
        errs = self.get_train_output() - outputs
        return errs.dot(errs) / numpy.alen(errs)
