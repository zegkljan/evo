# -*- coding: utf8 -*-
""" TODO docstring
"""

import copy

__author__ = 'Jan Žegklitz'


class Individual(object):
    """A class representing an individual in an evolutionary algorithm.

    This class is a "template" for other implementations of individuals. Best
    practice is to derive from this class, however, if a class of individual
    has the methods this class has, it will work.
    """

    def __init__(self):
        self.fitness = None
        self._data = dict()

    def set_fitness(self, fitness):
        """Sets the fitness of this individual.

        :param fitness: the fitness object
        """
        self.fitness = fitness

    def get_fitness(self):
        """Returns the fitness of this individual.

        It should return ``None`` if and only if the individual has not been
        evaluated yet.

        The base class implementation should not be modified if the objects
        passed to :meth:`.set_fitness()` are as described in that method doc.
        """
        return self.fitness

    def copy(self, carry_evaluation, carry_data):
        """Returns a copy of the individual.

        :param bool carry_evaluation: specifies whtether to copy the evaluation
            related data (most importantly the fitness value) too
        """
        raise NotImplementedError()

    def set_data(self, key, value):
        """Sets a data with the given key.
        """
        self._data[key] = value

    def get_data(self, key=None):
        """Returns the data under the given key or ``None`` if there is none.

        If the key is not specified (or set to ``None``) the whole dictionary
        will be returned.
        """
        if key is None:
            return self._data
        return self._data.get(key)

    @staticmethod
    def copy_evaluation(from_individual, to_individual, do_copy):
        """Copies the fitness value from `from_individual` to `to_individual` if
        `do_copy` is `True` (and does nothing if it is `False`).
        """
        if do_copy:
            to_individual.fitness = copy.deepcopy(from_individual.fitness)

    @staticmethod
    def copy_data(from_individual, to_individual, do_copy):
        """Copies the data from `from_individual` to `to_individual` if
        `do_copy` is `True` (and does nothing if it is `False`).
        """
        if do_copy:
            # noinspection PyProtectedMember
            to_individual._data = copy.deepcopy(from_individual._data)


class Fitness(object):
    """Takes care of evaluating individuals and assigns their fitness.

    This class is a "template" for other implementations of fitness evaluators.
    Best practice is to derive from this class, however, if a class of
    fitness evaluator has the methods this class has, it will work.

    The fitness is the most experiment-dependent part of evolutionary
    algorithms so it is almost a necessity to implement an fitness on a
    per-experiment basis.
    """

    COMPARE_TOURNAMENT = 'tournament'
    COMPARE_BSF = 'bsf'

    def evaluate(self, individual):
        """Evaluates the given individual and assigns the resulting fitness
        to the individual.

        The ``individual`` is expected to have methods ``get_fitness()`` and
        ``set_fitness()``.

        The evaluation is performed only if ``individual.get_fitness()``
        returns ``None``.

        :param individual: individual to be evaluated
        :see: :class:`.Individual`
        """
        raise NotImplementedError()

    def sort(self, population, reverse=False, *args):
        """Sorts ``population`` (which is expected to be a list of individuals)
        in an order that the best individual is the first and the worst the
        last.

        If ``reverse`` is ``True`` (default is ``False``) then the order is
        reversed (i.e. the worst is the first).

        :param args: possible additional arguments for comparison inside
            sorting. See :meth:`.compare`.

        :return: ``True`` if the population was successfully sorted,
            ``False`` if the population could not be sorted (e.g. for the
            nature of the fitness function).
        :rtype: bool
        """
        raise NotImplementedError()

    def compare(self, i1, i2, *args):
        """Returns ``True`` if individual ``i1`` is "better" than individual
        ``i2``.

        :param args: possible additional arguments for comparison. Can be
          used to distinguish multiple comparison types.
        """
        raise NotImplementedError()


class IndividualInitializer(object):
    """Base class for initializing individuals.

    Derive from this class to implement particular initializer.
    """

    def initialize(self):
        """Returns an initial individual.

        Override this method to implement your initialization mechanism.
        """
        raise NotImplementedError()


class PopulationInitializer(object):
    """Base class for initializing populations.

    Derive from this class to implement particular initializer.
    """

    def initialize(self, pop_size):
        """Returns an initial population.

        Override this method to implement your initialization mechanism.

        :param int pop_size: size of the population to initialize
        """
        raise NotImplementedError()


class SimplePopulationInitializer(PopulationInitializer):

    def __init__(self, individual_initializer):
        PopulationInitializer.__init__(self)
        self.individual_initializer = individual_initializer

    def initialize(self, pop_size):
        population = []
        for _ in range(pop_size):
            population.append(self.individual_initializer.initialize())
        return population


class GeneticBase(object):
    """A base class for genetic-like algorithm.

    This class contains the common utility methods only, not the algorithm
    itself.
    """
    def _pop_insert(self, indiv):
        """Inserts an individual into the sorted population.
        """
        if not self.population_sorted:
            raise ValueError('Population must be sorted.')

        # is it worse than the worst?
        if self.fitness.compare(self.population[-1], indiv,
                                Fitness.COMPARE_TOURNAMENT):
            self.population.append(indiv)
            return

        # is it better than the best?
        if self.fitness.compare(indiv, self.population[0],
                                Fitness.COMPARE_TOURNAMENT):
            self.population.insert(0, indiv)
            return

        # find the appropriate place by bisection
        l = 0
        u = len(self.population)
        c = (l + u) // 2
        while l < u and l != c != u:
            ci = self.population[c]
            if self.fitness.compare(ci, indiv, Fitness.COMPARE_TOURNAMENT):
                l = c
            elif self.fitness.compare(indiv, ci, Fitness.COMPARE_TOURNAMENT):
                u = c
            else:
                break
            c = (l + u) // 2
        self.population.insert(c + 1, indiv)

    def _pop_replace(self, replace_idx, indiv):
        """Removes the individual at ``replace_idx`` and inserts ``indiv``
        into the population.

        If the population is sorted the individual is inserted to the proper
        place. Otherwise the individual is placed at ``replace_idx``.
        """
        if not self.population_sorted:
            self.population[replace_idx] = indiv
            return

        # if the indiv fits to the place of replace_idx then put it there
        ln = None  # left neighbor
        rn = None  # right neighbor
        if replace_idx > 0:
            ln = self.population[replace_idx - 1]
        if replace_idx < len(self.population) - 1:
            rn = self.population[replace_idx + 1]

        left_fit = ln is not None and self.fitness.compare(ln, indiv,
                                                           Fitness.
                                                           COMPARE_TOURNAMENT)
        right_fit = rn is not None and self.fitness.compare(indiv, rn,
                                                            Fitness.
                                                            COMPARE_TOURNAMENT)
        if left_fit and right_fit:
            self.population[replace_idx] = indiv
            return

        # else just remove the individual at replace_idx and do a regular
        # insert to the population
        del self.population[replace_idx]
        self._pop_insert(indiv)