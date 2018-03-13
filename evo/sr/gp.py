# -*- coding: utf-8 -*-
"""TODO
"""

import logging

import numpy

import evo.gp.support
import evo.sr
import evo.utils.stats


class RegressionFitness(evo.Fitness):
    LOG = logging.getLogger(__name__ + '.RegressionFitness')

    def __init__(self, train_inputs, train_output, error_fitness,
                 handled_errors, stats: evo.utils.stats.Stats=None,
                 store_bsfs: bool=True,
                 fitness_measure: evo.sr.ErrorMeasure=evo.sr.ErrorMeasure.R2):
        super().__init__(store_bsfs)
        self.train_inputs = train_inputs
        self.train_output = numpy.array(train_output, copy=False)
        self.ssw = numpy.sum(
            (self.train_output - self.train_output.mean()) ** 2)
        self.error_fitness = error_fitness
        self.errors = tuple([evo.UnevaluableError] + handled_errors)
        self.stats = stats
        self.fitness_measure = fitness_measure

    def evaluate_individual(self, individual: evo.gp.support.ForestIndividual,
                            context=None):
        assert individual.genes_num == 1
        RegressionFitness.LOG.debug(
            'Evaluating individual %s in context %s', individual.__str__(),
            str(context))

        try:
            output = self.get_eval(individual, self.train_inputs)
            fitness = self.get_error(output, individual)
            individual.set_fitness(fitness)
        except self.errors as _:
            RegressionFitness.LOG.debug(
                'Exception occurred during evaluation, assigning fitness %f',
                self.error_fitness, exc_info=True)
            fitness = self.error_fitness
            individual.set_fitness(fitness)
        return individual.get_fitness()

    def compare(self, i1: evo.gp.support.ForestIndividual,
                i2: evo.gp.support.ForestIndividual, context=None):
        f1 = i1.get_fitness()
        f2 = i2.get_fitness()
        if f1 is None and f2 is not None:
            raise ValueError('First individual has no fitness.')
        if f1 is not None and f2 is None:
            raise ValueError('Second individual has no fitness.')
        if f1 is None and f2 is None:
            raise ValueError('Neither individual has fitness.')

        return self.fitness_cmp(f1, f2)

    def get_eval(self, individual: evo.gp.support.ForestIndividual,
                 args):
        return individual.genotype[0].eval(args=args)

    def get_error(self, output, individual: evo.gp.support.ForestIndividual):
        e = self.train_output - output
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
        if self.fitness_measure is evo.sr.ErrorMeasure.R2:
            return r2
        if self.fitness_measure is evo.sr.ErrorMeasure.MSE:
            return mse
        if self.fitness_measure is evo.sr.ErrorMeasure.MAE:
            return mae
        if self.fitness_measure is evo.sr.ErrorMeasure.WORST_CASE_AE:
            return worst_case_ae
        raise ValueError('Invalid value of fitness_measure.')

    def fitness_cmp(self, f1, f2):
        if self.fitness_measure is evo.sr.ErrorMeasure.R2:
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


def full_model_str(individual: evo.gp.support.ForestIndividual,
                   **kwargs) -> str:
    newline_genes = kwargs.get('newline_genes', False)
    strs = []
    for g in individual.genotype:
        strs.append('{}'.format(g.infix(**kwargs)))
    if newline_genes:
        return '\n+ '.join(strs)
    else:
        return ' + '.join(strs)
