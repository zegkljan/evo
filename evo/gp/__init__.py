# -*- coding: utf8 -*-
""" This package contains an implementation of classical Koza-style Genetic
Programming.
"""

import logging
import multiprocessing
import pprint
import gc

import random

import evo
import evo.gp.support
import evo.utils
import evo.utils.tree


# noinspection PyAbstractClass
class Gp(multiprocessing.context.Process):
    """This class forms the whole GE algorithm.
    """

    LOG = logging.getLogger(__name__ + '.Gp')

    class _GenerationsStop(object):

        def __init__(self, iterations):
            self.generations = iterations

        def __call__(self, gp):
            return gp.iterations >= self.generations

    def __init__(self, fitness, pop_strategy, selection_strategy,
                 population_initializer, functions, terminals, stop, name=None,
                 **kwargs):
        """The optional keyword argument ``generator`` can be used to pass a
        random number generator. If it is ``None`` or not present a standard
        generator is used which is the :mod:`random` module and its
        functions. If a generator is passed it is expected to have the
        methods corresponding to the :mod:`random` module (i.e. the class
        :class:`random.Random`).

        .. warning::

            The generator (does not matter whether a custom or the default
            one is used) is assumed that it is already seeded and no seed is
            set inside this class.

        :param evo.Fitness fitness: fitness used to evaluate individual
            performance
        :param pop_strategy: population handling strategy - determines the size
            of the (parent) population, the number of offspring in each
            iteration and how these should be combined with the parent
            population
        :type pop_strategy: :class:`evo.PopulationStrategy`
        :param selection_strategy: selection strategy - the algorithm of
            selection of "good" individuals from the population
        :type selection_strategy: :class:`evo.SelectionStrategy`
        :param population_initializer: initializer used to initialize the
            initial population
        :type population_initializer:
            :class:`ge.init.PopulationInitializer`
        :param functions: a list of functions available for the program
            synthesis
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param terminals: a list of terminals (i.e. zero-arity functions)
            available for the program synthesis
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param stop: Either a number or a callable. If it is number:

                The number of generations the algorithm will run for. One
                generation is when ``pop_size`` number of individuals were
                created and put back to the population. In other words,
                if the algorithm runs in generational mode then one
                generation is one iteration of the algorithm; if the
                algorithm runs in steady-state then one generation is half
                the ``pop_size`` iterations (because each iteration two
                individuals are selected, possibly crossed over and put back
                into the population).

            If it is a callable:

                The callable will be called at the beginning of each
                iteration of the algorithm with one argument which is the
                algorithm instance (i.e. instance of this class). If the
                return value is evaluated as ``True`` then the algorithm stops.
        :param str name: name of the process (see
            :class:`multiprocessing.Process`)
        :keyword generator: (keyword argument) a random number generator; if
            ``None`` or not present calls to the methods of standard python
            module :mod:`random` will be performed instead
        :type generator: :class:`random.Random` , or ``None``
        :keyword crossover_prob: (keyword argument) probability of performing a
            crossover; if it does not fit into interval [0, 1] it is set to 0 if
            lower than 0 and to 1 if higher than 1; default value is 0.8
        :keyword crossover_type: (keyword argument) the type of crossover;
            possible values are

                * ``'subtree'`` - subtree crossover

            The default value is ``'subtree'``.
        :keyword mutation_prob: (keyword argument) probability of performing
            a mutation; if it does not fit into interval [0, 1] it is set to
            0 if lower than 0 and to 1 if higher than 1; default value is 0.1
        :keyword mutation_type: (keyword argument) the type of mutation;
            possible values are

                * ``('subtree', max_depth)`` - subtree mutation; ``max_depth``
                  is the maximum depth of the randomly generated subtree; it is
                  the only mutation type supported

            The default value is ``('subtree', 5)``.
        :keyword evo.support.Stats stats: stats saving class
        :keyword callback: a callable which will be called at the beginning of
            every generation with a single argument which is the algorithm
            instance itself (i.e. instance of this class)
        """
        multiprocessing.context.Process.__init__(self, name=name)

        # Positional args
        self.fitness = fitness
        self.pop_strategy = pop_strategy
        self.selection_strategy = selection_strategy
        self.functions = functions
        self.terminals = terminals
        self.population_initializer = population_initializer
        if isinstance(stop, int):
            # noinspection PyProtectedMember
            self.stop = Gp._GenerationsStop(stop)
        elif callable(stop):
            self.stop = stop
        else:
            raise TypeError('Argument stop is neither integer nor callable.')

        # Keyword args
        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.crossover_prob = 0.8
        if 'crossover_prob' in kwargs:
            self.crossover_prob = kwargs['crossover_prob']
            self.crossover_prob = max(0, self.crossover_prob)
            self.crossover_prob = min(1, self.crossover_prob)

        self.crossover_method = self.subtree_crossover
        self.crossover_method_args = ()
        if 'crossover_type' in kwargs and kwargs['crossover_type'] is not None:
            self.crossover_method, self.crossover_method_args = \
                self.setup_crossover(kwargs['crossover_type'])

        self.mutation_prob = 0.1
        if 'mutation_prob' in kwargs:
            self.mutation_prob = kwargs['mutation_prob']
            self.mutation_prob = max(0, self.mutation_prob)
            self.mutation_prob = min(1, self.mutation_prob)

        self.mutate_method = self.subtree_mutate
        self.mutate_method_args = (5,)
        if 'mutation_type' in kwargs and kwargs['mutation_type'] is not None:
            self.mutate_method, self.mutate_method_args = \
                self.setup_mutation(kwargs['mutation_type'])

        self.stats = None
        if 'stats' in kwargs:
            self.stats = kwargs['stats']

        self.callback = None
        if 'callback' in kwargs:
            self.callback = kwargs['callback']
            if not callable(self.callback):
                raise TypeError('Keyword argument callback is not a callable.')

        self.population = []
        self.population_sorted = False

        self.iterations = 0
        """
        The number of elapsed iterations of the algorithm (either generations
        in the generational mode or just iterations in the steady-state mode).
        """

    def run(self):
        """Runs the GE algorithm.
        """
        Gp.LOG.info('Starting algorithm.')
        try:
            self.population = self.population_initializer.initialize(
                self.pop_strategy.get_parents_number())

            self._run()
        finally:
            if self.fitness.get_bsf() is None:
                Gp.LOG.info('Finished. No BSF acquired.')
            else:
                Gp.LOG.info('Finished.\nFitness: %f\n%s',
                            self.fitness.get_bsf().get_fitness(),
                            pprint.pformat(self.fitness.get_bsf().get_data()))
            Gp.LOG.info('Performing garbage collection.')
            gc.collect()
            try:
                if self.stats is not None:
                    self.stats.cleanup()
            except AttributeError:
                pass

    def _run(self):
        Gp.LOG.info('Starting evolution.')
        while not self.stop(self):
            Gp.LOG.info('Starting iteration %d', self.iterations)
            if self.callback is not None:
                self.callback(self)

            elites = self.top_individuals(self.pop_strategy.get_elites_number())

            Gp.LOG.debug('Processing selection.')
            offspring = []
            while len(offspring) < self.pop_strategy.get_offspring_number():
                a = self.selection_strategy.select_single(self.population)[1]
                if self.generator.random() < self.crossover_prob:
                    b = self.selection_strategy.select_single(
                        self.population)[1]
                    children = self.crossover(a.copy(), b.copy())
                else:
                    children = [a.copy()]

                while (children and
                       len(offspring) <
                        self.pop_strategy.get_offspring_number()):
                    o = children.pop()
                    if self.generator.random() < self.mutation_prob:
                        o = self.mutate(o)
                    offspring.append(o)
            self.population = self.pop_strategy.combine_populations(
                self.population, offspring, elites)
            Gp.LOG.info('Finished iteration %d', self.iterations)
            self.iterations += 1
        if self.callback is not None:
            self.callback(self)
        Gp.LOG.info('Finished evolution.')

    def setup_crossover(self, crossover_type):
        """Helper method for the constructor which sets up the crossover method.
        """
        if crossover_type == 'subtree':
            crossover_method = self.subtree_crossover
            crossover_method_args = ()
        else:
            raise ValueError('Invalid crossover type.')
        return crossover_method, crossover_method_args

    def setup_mutation(self, mutation_type):
        """Helper method for the constructor which sets up the mutation method.
        """
        if mutation_type[0] == 'subtree':
            mutation_method = self.subtree_mutate
            mutation_method_args = (mutation_type[1],)
        else:
            raise ValueError('Invalid crossover type.')
        return mutation_method, mutation_method_args

    def crossover(self, o1, o2):
        """Performs a crossover of two individuals.

        :param evo.gp.support.TreeIndividual o1: first parent
        :param evo.gp.support.TreeIndividual o2: second parent
        """
        Gp.LOG.debug('Performing crossover of individuals %s, %s', o1, o2)
        assert self.crossover_method is not None
        # noinspection PyArgumentList
        return self.crossover_method(o1, o2, *self.crossover_method_args)

    def mutate(self, i):
        """Performs a mutation of the individual.

        The individual is mutated in place and also returned.

        :param evo.gp.support.TreeIndividual i: the individual to be mutated
        :return: the mutated individual
        :rtype: :class:`evo.gp.support.TreeIndividual`
        """
        return self.mutate_method(i, *self.mutate_method_args)

    def subtree_crossover(self, o1, o2):
        g1 = o1.genotype
        g2 = o2.genotype

        s1 = g1.get_subtree_size()
        s2 = g2.get_subtree_size()

        p1 = self.generator.randrange(s1)
        p2 = self.generator.randrange(s2)

        n1 = g1.get_nth_node(p1)
        n2 = g2.get_nth_node(p2)

        r1, r2 = evo.gp.support.swap_subtrees(n1, n2)
        o1.genotype = r1
        o2.genotype = r2

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]

    def subtree_mutate(self, i, max_depth):
        g = i.genotype

        s = g.get_subtree_size()
        p = self.generator.randrange(s)
        n = g.get_nth_node(p)

        subtree = evo.gp.support.generate_grow(self.functions, self.terminals,
                                               max_depth, self.generator)
        i.genotype = evo.gp.support.replace_subtree(n, subtree)
        i.set_fitness(None)
        return i

    def top_individuals(self, k):
        if k <= 0:
            return []
        kth = evo.utils.select(self.population, k - 1,
                               cmp=lambda a, b: self.fitness.is_better(a, b))
        tops = []
        for i in self.population:
            if self.fitness.is_better(i, kth) or self.fitness.is_better(kth, i):
                tops.append(i)
            if len(tops) == k:
                break
        return tops
