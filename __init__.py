# -*- coding: utf8 -*-
import random


class Individual(object):
    """A class representing an individual in an evolutionary algorithm.

    This class is a "template" for other implementations of individuals. Best
    practice is to derive from this class, however, if a class of individual
    has the methods this class has, it will work.
    """

    def __init__(self):
        self.fitness = None

    def set_fitness(self, fitness):
        """Sets the fitness of this individual.

        :param dict fitness: the fitness object. It should be a :class:`dict`
            with an entry with key ``'fitness'`` whose value is the
            individual's fitness."""
        self.fitness = fitness

    def get_fitness(self):
        """Returns the fitness of this individual.

        It should return ``None`` if and only if the individual has not been
        evaluated yet.

        The base class implementation should not be modified if the objects
        passed to :meth:`.set_fitness()` are as described in that method doc.
        """
        if self.fitness is None:
            return None
        return self.fitness['fitness']

    def get_fitness_object(self):
        """Returns the whole fitness object.

        The whole object is returned, not only its the ``'fitness'`` field.
        """
        return self.fitness

    def copy(self, carry_evaluation):
        """Returns a copy of the individual.

        :param bool carry_evaluation: specifies whtether to copy the evaluation
            related data (most importantly the fitness value) too
        """
        pass

    def __str__(self):
        return self.get_fitness()


class Fitness(object):
    """Takes care of evaluating individuals and assigns their fitness.

    This class is a "template" for other implementations of fitness evaluators.
    Best practice is to derive from this class, however, if a class of
    fitness evaluator has the methods this class has, it will work.

    The fitness is the most experiment-dependent part of evolutionary
    algorithms so it is almost a necessity to implement an fitness on a
    per-experiment basis.
    """

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
        pass

    def sort(self, population, reverse=False):
        """Sorts ``population`` (which is expected to be a list of individuals)
        in an order that the best individual is the first and the worst the
        last.

        If ``reverse`` is ``True`` (default is ``False``) then the order is
        reversed (i.e. the worst is the first).
        """
        pass

    def compare(self, f1, f2):
        """Returns ``True`` if fitness ``f1`` is "better" than fitness ``f2``.
        """
        pass


class IndividualInitializer(object):
    """Base class for initializing individuals.

    Derive from this class to implement particular initializer.

    :see: :class:`.RandomCodonGenotypeInitializer`
    """

    def __init__(self):
        pass

    def initialize(self):
        """Returns an initial individual.

        Override this method to implement your initialization mechanism.
        """
        pass


class PopulationInitializer(object):
    """Base class for initializing populations.

    Derive from this class to implement particular initializer.
    """

    def __init__(self):
        pass

    def initialize(self, pop_size):
        """Returns an initial population.

        Override this method to implement your initialization mechanism.

        :param int pop_size: size of the population to initialize
        """
        pass


class SimplePopulationInitializer(PopulationInitializer):

    def __init__(self, individual_initializer):
        PopulationInitializer.__init__(self)
        self.individual_initializer = individual_initializer

    def initialize(self, pop_size):
        population = []
        for _ in range(pop_size):
            population.append(self.individual_initializer.initialize())
        return population
