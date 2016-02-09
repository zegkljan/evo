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
class RpropFitness(evo.Fitness):
    """A fitness for symbolic regression that operates with solutions made of
    trees that support backpropagation.
    """

    LOG = logging.getLogger(__name__ + '.BackpropagationFitness')

    def __init__(self, error_fitness, handled_errors, train_inputs,
                 train_output, var_mapping: dict, rprop_updater: callable,
                 rprop_steps=10):
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
        :param train_inputs: feature variables' values: an N x M matrix where N
            is the number of datapoints (the same N as in ``target`` argument)
            and M is the number of feature variables
        :param train_output: target values of the datapoints: an N x 1 matrix
            where N is the number of datapoints
        :param var_mapping: mapping from :class:`int` to :class:`str` that maps
            a number of an input variable to variable name (the number of
            feature is the number of corresponding column in the
            ``train_inputs`` argument)
        :param rprop_steps: number of steps the RPROP optimisation algorithm
            will do prior to each error evaluation
        """
        super().__init__()
        self.error_fitness = error_fitness
        self.train_inputs = train_inputs
        self.train_output = numpy.array(train_output, copy=False)
        self.var_mapping = var_mapping
        self.errors = tuple([ZeroDivisionError, FloatingPointError] +
                            handled_errors)
        self.rprop_updater = rprop_updater
        self.rprop_steps = rprop_steps

        self.args = evo.sr.prepare_args(train_inputs, var_mapping)
        self.cost_derivative = lambda yhat, y: yhat - y
        self.bsf = None

    def evaluate(self, individual: evo.gp.support.TreeIndividual):
        RpropFitness.LOG.debug('Evaluating individual %s',
                               individual.__str__())

        RpropFitness.LOG.debug('Optimising inner parameters.')
        try:
            yhat = individual.genotype.eval(args=self.args)
            error = prev_error = self.analyze_error(self.train_output - yhat,
                                                    individual)
            RpropFitness.LOG.debug('Initial error: %f', error)
            for i in range(self.rprop_steps):
                evo.sr.backpropagation.backpropagate(individual.genotype,
                                                     self.cost_derivative,
                                                     self.train_output,
                                                     self.args,
                                                     self.train_inputs.shape[0])
                updated = self.rprop_updater.update(individual.genotype,
                                                    error, prev_error)

                if not updated:
                    RpropFitness.LOG.debug('Stopping rprop because no update '
                                           'occurred.')
                    break
                yhat = individual.genotype.eval(args=self.args)
                if numpy.any(numpy.logical_or(numpy.isinf(yhat),
                                              numpy.isnan(yhat))):
                    RpropFitness.LOG.info('NaN or inf in output, assigning '
                                          'fitness: %f',
                                          self.error_fitness)
                    return self.error_fitness
                prev_error = error
                error = self.analyze_error(self.train_output - yhat, individual)
                RpropFitness.LOG.debug('Step %d error: %f Full model: %s', i,
                                       error, individual.genotype.full_infix())
                pass
            error = self.train_output - yhat
            fitness = self.analyze_error(error, individual)
        except self.errors as e:
            RpropFitness.LOG.debug('Exception occurred during evaluation, '
                                   'assigning fitness %f',
                                   self.error_fitness,
                                   exc_info=True)
            fitness = self.error_fitness
        individual.set_fitness(fitness)
        if self.bsf is None or self.bsf.get_fitness() > fitness:
            self.bsf = individual.copy()
        return fitness

# noinspection PyMethodMayBeStatic,PyUnusedLocal
    def analyze_error(self, error, individual):
        """Computes the fitness value from the error on target data.

        Computes the fitness value as the sum of squared errors. Override this
        method to use different fitness and/or to perform additional analysis
        of error values.

        :param error: a vector of errors
        :param individual: the individual whose errors are analyzed
        :return: the resulting fitness value
        """
        return error.T.dot(error) / numpy.alen(error)

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
            f2 = i1.get_fitness()
        return f1 < f2

    def get_bsf(self) -> evo.Individual:
        return self.bsf
