# -*- coding: utf-8 -*-
""" TODO docstring
"""

import copy
import logging
from builtins import round

__version__ = '0.1'


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

        :param bool carry_evaluation: specifies whether to copy the evaluation
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

    def delete_data(self, key):
        """Deletes the data under the given key.

        If there is no such key, nothing happens.
        """
        if key in self._data:
            del self._data[key]

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

    def __init__(self, store_bsfs: bool=True):
        self.store_bsfs = store_bsfs

        self.bsf = None
        self.bsfs = []

    def evaluate(self, individual, context=None):
        """Evaluates the given individual, checkes whether it is a new bsf or
        not and if it is it stores it as one.

        The method effectively calls methods :meth:`.evaluate_individual` and
        :meth:`handle_bsf` in sequence.

        .. seealso:: :meth:`.evaluate_individual` and :meth:`handle_bsf`
        """
        self.evaluate_individual(individual, context)
        self.handle_bsf(individual, context)

    def evaluate_individual(self, individual: Individual, context=None):
        """Evaluates the given individual and assigns the resulting fitness
        to the individual.

        The ``individual`` is expected to have methods ``get_fitness()`` and
        ``set_fitness()``.

        The evaluation is performed only if ``individual.get_fitness()``
        returns ``None``.

        :param individual: individual to be evaluated
        :param context: arbitrary data an algorithm can provide to the fitness
            (e.g. iteration number)

        .. seealso:: :class:`.Individual`
        """
        raise NotImplementedError()

    def handle_bsf(self, individual: Individual, context=None,
                   do_not_copy: bool=False):
        """Checks whether the given individual is best-so-far (bsf) and if it
        is it stores it.

        The bsf can be retrieved via the :meth:`.get_bsf` method.

        The individual is always copied via its :meth:`evo.Individual.copy`
        method unles turned off by setting the argument *do_not_copy* to
        ``True``\ .

        :param individual: individual to be checked for being bsf
        :param context: arbitrary data an algorithm can provide to the fitness
            (e.g. iteration number)
        :param do_not_copy: if ``True`` the bsf is stored as the individual
            itself and not its copy

        .. seealso:: :meth:`.get_bsf`
        """
        if self.bsf is None or self.compare(individual, self.bsf) < 0:
            if do_not_copy:
                self.bsf = individual
            else:
                self.bsf = individual.copy()

            if self.store_bsfs:
                self.bsfs.append(self.bsf)

    def compare(self, i1, i2, context=None):
        """Returns ``-1`` if individual ``i1`` is strictly better than
        individual ``i2``, ``0`` if they are of equal quality and ``1`` if
        ``i1`` is strictly worse than ``i2``.

        :param context: arbitrary data an algorithm can provide to the fitness
            (e.g. iteration number)
        """
        raise NotImplementedError()

    def is_better(self, i1, i2, context=None):
        """A wrapper for :meth:`evo.Fitness.compare` simplifying the output to
        a boolean value which is ``True`` only if ``i1`` is strictly better than
        ``i2``.

        All arguments have exactly the same meaning as in
        :meth:`evo.Fitness.compare`.
        """
        return self.compare(i1, i2, context) < 0

    def get_bsf(self) -> Individual:
        """Returns the best solution encountered so far.

        :return: the best-so-far solution or ``None`` if there is no such
            solution (yet)
        :rtype: :class:`evo.Individual`
        """
        return self.bsf

    def get_bsfs(self):
        """Returns a list of the best-so-far solutions in order as they were
        found during the optimisation run.
        """
        return self.bsfs


class UnevaluableError(Exception):
    pass


class StopEvolution(Exception):
    def __init__(self, reason):
        self.reason = reason

    def __str__(self):
        return 'Evolution stopped. Reason: {}'.format(self.reason)

    def __repr__(self):
        return 'StopEvolution({})'.format(repr(self.reason))


class IndividualInitializer(object):
    """Base class for initializing individuals.

    Derive from this class to implement particular initializer.
    """

    def initialize(self, limits: dict):
        """Returns an initial individual.

        Override this method to implement your initialization mechanism.

        :param dict limits: dictionary with limits imposed on the individuals by
            the caller
        """
        raise NotImplementedError()


class PopulationInitializer(object):
    """Base class for initializing populations.

    Derive from this class to implement particular initializer.
    """

    def initialize(self, pop_size: int, limits: dict):
        """Returns an initial population.

        Override this method to implement your initialization mechanism.

        :param int pop_size: size of the population to initialize
        :param dict limits: dictionary with limits imposed on the individuals by
            the caller
        """
        raise NotImplementedError()


class SimplePopulationInitializer(PopulationInitializer):
    LOG = logging.getLogger(__name__ + '.SimplePopulationInitializer')

    def __init__(self, individual_initializer: IndividualInitializer):
        PopulationInitializer.__init__(self)
        self.individual_initializer = individual_initializer

    def initialize(self, pop_size: int, limits: dict=None):
        SimplePopulationInitializer.LOG.info('Initializing population of size '
                                             '%d', pop_size)
        population = []
        for _ in range(pop_size):
            population.append(self.individual_initializer.initialize(limits))
        SimplePopulationInitializer.LOG.info('Population initialized.')
        return population


class Evolution(object):
    """A base class for genetic-like algorithm.

    This class contains the common utility methods only, not the algorithm
    itself.
    """

    def compare_individuals(self, a, b) -> int:
        """Compares two individuals and returns ``-1`` if the first one is
        better than the second one, ``0`` if they are equally good and ``1`` if
        the first one is worse than the second one."""
        raise NotImplementedError()


class PopulationStrategy(object):
    """Defines the population dynamics in a genetic (evolutionary) algorithm.

    Objects of this class are responsible for determining the size of the
    population, the number of offspring generated in each iteration and how
    these offspring are combined with the (parent) population.
    """

    def get_parents_number(self) -> int:
        """Returns the number of individuals in the parent population.

        In classical GA this is equivalent to the population size.

        :rtype: :class:`int`
        """
        raise NotImplementedError()

    def get_offspring_number(self) -> int:
        """Returns the number of individuals created in each iteration.

        :rtype: :class:`int`
        """
        raise NotImplementedError()

    def get_elites_number(self) -> int:
        """Returns the number of elite individuals that are to be directly
        copied to the next iteration.

        :rtype: :class:`int`
        """
        raise NotImplementedError()

    def combine_populations(self, parents, offspring, elites):
        """Combines the parent population, population of offspring and the
        elites to create the parent population for the next iteration.

        :param parents: parent population without the elites
        :type parents: :class:`list` of :class:`evo.Individual`
        :param offspring: population of offspring
        :type offspring: :class:`list` of :class:`evo.Individual`
        :param elites: elites (the best individuals from parent population)
        :type elites: :class:`list` of :class:`evo.Individual`
        :return: a new population to serve as a parent population in the next
            iteration
        :rtype: :class:`list` of :class:`evo.Individual`
        """
        raise NotImplementedError()


class GenerationalPopulationStrategy(PopulationStrategy):
    """Handles the generational strategy:

    The population is of size N

        #. E top individuals (i.e. elites) are extracted from parent population
        #. (N - E) offspring are created
        #. offspring and elites are joined into a single population which
           completely replaces the parent population
    """

    def __init__(self, pop_size, elites_num):
        """
        :param pop_size: (parent) population size
        :param elites_num: number of elites; if it is a float form range
            [0, 1) then it is considered as a fraction of the ``pop_size``
            and that fraction is going to be used as the number of elites
            (after rounding)
        """
        self.pop_size = pop_size
        self.elites_num = elites_num
        if 0 <= elites_num < 1:
            self.elites_num = int(round(pop_size * elites_num))

    def get_elites_number(self):
        return self.elites_num

    def get_parents_number(self):
        return self.pop_size

    def get_offspring_number(self):
        return self.pop_size - self.elites_num

    def combine_populations(self, parents, offspring, elites):
        return elites + offspring


class SteadyStatePopulationStrategy(PopulationStrategy):
    """Handles the steady-state strategy:

    The population is of size N

        #. X (X is much lower than N) offspring are created
        #. offspring are put back into population, killing some individuals
           depending on the :meth:`.replace` method

    There are no explicit elites.
    """

    def __init__(self, pop_size, offspring_num):
        """
        :param pop_size: (parent) population size
        :param offspring_num: number of offspring to be created each iteration
        """
        self.pop_size = pop_size
        self.offspring_num = offspring_num

    def get_elites_number(self):
        return 0

    def get_parents_number(self):
        return self.pop_size

    def get_offspring_number(self):
        return self.offspring_num

    def combine_populations(self, parents, offspring, elites):
        for o in offspring:
            self.replace(parents, o)
        return parents + elites

    def replace(self, parent_pop, individual):
        """Puts an individual into (parent) population such that the population
        size remains the same (i.e. some individual has to be thrown away)

        :param parent_pop: (parent) population
        :type parent_pop: :class:`list` of :class:`evo.Individual`
        :param evo.Individual individual: an individual to be put into the
            population
        :rtype: :class:`list` of :class:`evo.Individual`
        """
        raise NotImplementedError()


class SelectionStrategy(object):
    """Defines the selection algorithm (strategy).
    """

    def select_single(self, population, algorithm):
        """Selects a single individual from the given population.

        :param population: population
        :type population: :class:`list` of :class:`evo.Individual`
        :param algorithm: the object corresponding to the algorithm that runs
            this selection
        :type algorithm: :class:`evo.Evolution`
        :return: a tuple containing the selected individual as the first element
            and its index in the population as the second element
        :rtype: :class:`tuple` of :class:`evo.Individual` and :class:`int`
        """
        raise NotImplementedError()

    def select_all(self, population, all_num, algorithm):
        """Selects all individuals (up to the given number) from the given
        population.

        .. note::

            Default implementation is::

                def select_all(self, population, all_num):
                    out = []
                    for i in range(all_num):
                        out.append(self.select_single(population))
                    return out

        :param population: population
        :type population: :class:`list` of :class:`evo.Individual`
        :param int all_num: number of individuals to be selected
        :param algorithm: the object corresponding to the algorithm that runs
            this selection
        :type algorithm: :class:`evo.Evolution`
        :return: a list of tuples, each containing a selected individual as the
            first element and its index in the population as the second element
        :rtype: :class:`list` of :class:`tuple` of :class:`evo.Individual` and
            :class:`int`
        """
        out = []
        for i in range(all_num):
            out.append(self.select_single(population, algorithm))
        return out


class TournamentSelectionStrategy(SelectionStrategy):
    """Handles the tournament selection of individuals.
    """

    def __init__(self, tournament_size, generator):
        """
        :param int tournament_size: size of the tournament
        :param random.Random generator: random number generator
        """
        if tournament_size < 2:
            raise ValueError("Tournament size must be at least 2.")
        self.tournament_size = tournament_size
        self.generator = generator

    def select_single(self, population, algorithm: Evolution):
        best_idx = self.generator.randrange(len(population))
        for _ in range(self.tournament_size - 1):
            idx = self.generator.randrange(len(population))
            if algorithm.compare_individuals(population[idx],
                                             population[best_idx]) < 0:
                best_idx = idx
        return best_idx, population[best_idx]


class ReproductionStrategy(object):
    """Defines how are the offspring created from the parents.
    """

    def reproduce(self,
                  selection_strategy: SelectionStrategy,
                  population_strategy: PopulationStrategy,
                  algorithm: Evolution,
                  parents, offspring):
        """Produces one or more offspring based on the list of potential parents
        and inserts them to the ``offspring`` list.

        This method encapsulates a *single* reproduction event. That means that
        this method can be called repeatedly by the driver algorithm.

        :param selection_strategy: a selection strategy
        :param population_strategy: a population strategy
        :param algorithm: the object corresponding to the algorithm that runs
            this reproduction
        :param parents: list of individuals that can be the parents
        :type parents: :class:`list` of :class:`evo.Individual`
        :param offspring: list of offspring individuals
        :type offspring: :class:`list` of :class:`evo.Individual`
        """
        raise NotImplementedError()
