# -*- coding: utf8 -*-
""" TODO docstring
"""

import logging

import copy


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

        .. seealso:: :class:`.Individual`
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
        """Returns ``-1`` if individual ``i1`` is strictly better than
        individual ``i2``, ``0`` if they are of equal quality and ``1`` if
        ``i1`` is strictly worse than ``i2``.

        :param args: possible additional arguments for comparison. Can be
          used to distinguish multiple comparison types.
        """
        raise NotImplementedError()

    def is_better(self, i1, i2, *args):
        """A wrapper for :meth:`evo.Fitness.compare` simplifying the output to
        a boolean value which is ``True`` only if ``i1`` is strictly better than
        ``i2``.

        All arguments have exactly the same meaning as in
        :meth:`evo.Fitness.compare`.
        """
        return self.compare(i1, i2, *args) <= 0

    def get_bsf(self) -> Individual:
        """Returns the best solution encountered so far.

        :return: the best-so-far solution or ``None`` if there is no such
            solution (yet)
        :rtype: :class:`evo.Individual`
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
    LOG = logging.getLogger(__name__ + '.SimplePopulationInitializer')

    def __init__(self, individual_initializer):
        PopulationInitializer.__init__(self)
        self.individual_initializer = individual_initializer

    def initialize(self, pop_size):
        SimplePopulationInitializer.LOG.info('Initializing population of size '
                                             '%d', pop_size)
        population = []
        for _ in range(pop_size):
            population.append(self.individual_initializer.initialize())
        SimplePopulationInitializer.LOG.info('Population initialized.')
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
        if self.fitness.is_better(self.population[-1], indiv,
                                  Fitness.COMPARE_TOURNAMENT):
            self.population.append(indiv)
            return

        # is it better than the best?
        if self.fitness.is_better(indiv, self.population[0],
                                  Fitness.COMPARE_TOURNAMENT):
            self.population.insert(0, indiv)
            return

        # find the appropriate place by bisection
        l = 0
        u = len(self.population)
        c = (l + u) // 2
        while l < u and l != c != u:
            ci = self.population[c]
            if self.fitness.is_better(ci, indiv, Fitness.COMPARE_TOURNAMENT):
                l = c
            elif self.fitness.is_better(indiv, ci, Fitness.COMPARE_TOURNAMENT):
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

        left_fit = ln is not None and self.fitness.is_better(ln, indiv,
                                                             Fitness.
                                                             COMPARE_TOURNAMENT)
        right_fit = rn is not None and self.fitness.is_better(
            indiv, rn, Fitness.COMPARE_TOURNAMENT)
        if left_fit and right_fit:
            self.population[replace_idx] = indiv
            return

        # else just remove the individual at replace_idx and do a regular
        # insert to the population
        del self.population[replace_idx]
        self._pop_insert(indiv)


class PopulationStrategy(object):
    """Defines the population dynamics in a genetic (evolutionary) algorithm.

    Objects of this class are responsible for determining the size of the
    population, the number of offspring generated in each iteration and how
    these offspring are combined with the (parent) population.
    """

    def get_parents_number(self):
        """Returns the number of individuals in the parent population.

        In classical GA this is equivalent to the population size.

        :rtype: :class:`int`
        """
        raise NotImplementedError()

    def get_offspring_number(self):
        """Returns the number of individuals created in each iteration.

        :rtype: :class:`int`
        """
        raise NotImplementedError()

    def get_elites_number(self):
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
        :param elites_num: number of elites
        """
        self.pop_size = pop_size
        self.elites_num = elites_num

    def get_elites_number(self):
        return self.elites_num

    def get_parents_number(self):
        return self.pop_size

    def get_offspring_number(self):
        return self.pop_size - self.elites_num

    def combine_populations(self, parents, offspring, elites):
        return offspring + elites


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

    def select_single(self, population):
        """Selects a single individual from the given population.

        :param population: population
        :type population: :class:`list` of :class:`evo.Individual`
        :return: a tuple containing the selected individual as the first element
            and its index in the population as the second element
        :rtype: :class:`tuple` of :class:`evo.Individual` and :class:`int`
        """
        raise NotImplementedError()

    def select_all(self, population, all_num):
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
        :return: a list of tuples, each containing a selected individual as the
            first element and its index in the population as the second element
        :rtype: :class:`list` of :class:`tuple` of :class:`evo.Individual` and
            :class:`int`
        """
        out = []
        for i in range(all_num):
            out.append(self.select_single(population))
        return out


class TournamentSelectionStrategy(SelectionStrategy):
    """Handles the tournament selection of individuals.
    """

    def __init__(self, tournament_size, generator, fitness):
        """
        :param int tournament_size: size of the tournament
        :param random.Random generator: random number generator
        :param evo.Fitness fitness: the fitness object for comparison of
            individuals
        """
        if tournament_size < 2:
            raise ValueError("Tournament size must be at least 2.")
        self.tournament_size = tournament_size
        self.generator = generator
        self.fitness = fitness

    def select_single(self, population):
        best_idx = self.generator.randrange(len(population))
        for _ in range(self.tournament_size - 1):
            idx = self.generator.randrange(len(population))
            if self.fitness.is_better(population[idx], population[best_idx]):
                best_idx = idx
        return best_idx, population[best_idx]


if __name__ == '__main__':
    print('Run')
