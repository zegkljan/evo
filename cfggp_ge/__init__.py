# -*- coding: utf8 -*-
"""An attepmt to mix cfg-gp with ge.
"""
import fractions
import functools
import random
import multiprocessing.context

import wopt.evo
import wopt.evo.cfggp
import wopt.evo.ge
import wopt.evo.support.grammar
import wopt.utils


class DerivationToCodonsPopulationInitializer(wopt.evo.PopulationInitializer):

    def __init__(self, grammar, **kwargs):
        wopt.evo.PopulationInitializer.__init__(self)
        self.grammar = grammar
        self.derivation_tree_population = []

        if 'generator' in kwargs:
            self.generator = kwargs['generator']
        else:
            self.generator = random

        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']
        else:
            self.multiplier = 1

        choice_nums = [r.get_choices_num() for r in grammar.get_rules()]
        m = functools.reduce(lambda a, b: a * b // fractions.gcd(a, b),
                             choice_nums)
        self.max_choices = m * self.multiplier

    def initialize(self, pop_size):
        assert pop_size == len(self.derivation_tree_population)
        population = []
        for i in self.derivation_tree_population:
            population.append(self.derivation_to_codons(i))
        return population

    def derivation_to_codons(self, individual):
        if isinstance(individual, wopt.evo.ge.CodonGenotypeIndividual):
            return individual
        if not isinstance(individual, wopt.evo.cfggp.DerivationTreeIndividual):
            raise TypeError('Individual must be of type '
                            'wopt.evo.cfggp.DerivationTreeIndividual.')

        choices, max_choices = \
            self.grammar.derivation_tree_to_choice_sequence(individual.tree)
        genotype = self.randomize(choices, max_choices)
        assert (self.grammar.generate(genotype, mode='tree')[0].__str__() ==
                individual.tree.__str__())
        return wopt.evo.ge.CodonGenotypeIndividual(genotype, self.max_choices)

    def randomize(self, choices, max_choices):
        randomized = []
        for choice, choices_num in zip(choices, max_choices):
            limit = self.max_choices // choices_num
            rnd = self.generator.randrange(limit) * choices_num
            randomized.append(choice + rnd)
        return randomized


class CfggpGe(multiprocessing.context.Process):
    """This class forms the combination of CFG-GP and GE.
    """

    class _StagnationDetector(object):

        def __init__(self, flat_generations, conventional_stop):
            object.__init__(self)
            self.flat_generations_limit = flat_generations
            self.last_bsf_fitness = None
            self.conventional_stop = conventional_stop
            self.last_bsf_generations = 0

        def __call__(self, cfggp):
            if self.conventional_stop(cfggp):
                return True
            if (self.last_bsf_fitness is None and cfggp.bsf is not None and
                cfggp.bsf.get_fitness() is not None):
                self.last_bsf_fitness = cfggp.bsf.get_fitness()
                return False
            if cfggp.bsf is None or cfggp.bsf.get_fitness() is None:
                return False
            if cfggp.fitness.compare(cfggp.bsf.get_fitness(),
                                     self.last_bsf_fitness):
                self.last_bsf_fitness = cfggp.bsf.get_fitness()
                self.last_bsf_generations = cfggp.iterations
                return False
            else:
                if cfggp.mode == 'generational':
                    return cfggp.iterations - self.last_bsf_generations >= \
                        self.flat_generations_limit
                if cfggp.mode == 'steady-state':
                    return cfggp.iterations - self.last_bsf_generations >= (
                        self.flat_generations_limit * cfggp.pop_size)
                assert False

    class _GenerationsStop(object):

        def __init__(self, generations):
            self.generations = generations

        def __call__(self, runner):
            if runner.mode == 'generational':
                return runner.iterations >= self.generations
            if runner.mode == 'steady-state':
                return runner.iterations >= (self.generations * runner.pop_size)

    class _StaticInitializer(wopt.evo.PopulationInitializer):

        def __init__(self):
            wopt.evo.PopulationInitializer.__init__(self)
            self.population = []

        def set_population(self, population):
            self.population = population

        def initialize(self, pop_size):
            assert pop_size == len(self.population)
            return self.population

    def __init__(self, fitness, pop_size, population_initializer, grammar, mode,
                 stop, stagnation_trigger, name=None, **kwargs):
        """The optional keyword argument ``generator`` can be used to pass a
        random number generator. If it is ``None`` or not present a standard
        generator is used which is the :mod:`random` module and its
        functions. If a generator is passed it is expected to have the
        methods corresponding to the :mod:`random` module (individual.e. the
        class :class:`random.Random`).

        .. warning::

            The generator (does not matter whether a custom or the default
            one is used) is assumed that it is already seeded and no seed is
            set inside this class.

        :param fitness: fitness used to evaluate individual performance
        :type fitness: :class:`wopt.evo.Fitness`
        :param int pop_size: size of the population; this value will be
            passed to the ``population_initializer``'s method ``initialize``()
        :param population_initializer: initializer used to initialize the
            initial population
        :type population_initializer:
            :class:`wopt.ge.init.PopulationInitializer`
        :param grammar: grammar this algorithm operates on
        :type grammar: :class:`wopt.evo.support.grammar.Grammar`
        :param mode: Specifies which mode of genetic algorithm to use. Possible
            values are ``'generational'`` and ``'steady-state'``.
        :param stop: Either a number or a callable. If it is number:

                The number of generations the algorithm will run. One
                generation is when ``pop_size`` number of individuals were
                created and put back to the population. In other words,
                if the algorithm runs in generational mode then one
                generation is one iteration of the algorithm; if the
                algorithm runs in steady-state then none generation is half
                the ``pop_size`` iterations (because each iteration two
                individuals are selected, possibly crossed over and put back
                into the population.

            If it is a callable:

                The callable will be called at the beginning of each
                iteration of the algorithm with one argument which is the
                algorithm instance (i.e. instance of this class). If the
                return value is evaluated as ``True`` then the algorithm stops.
        :param int stagnation_trigger: a number of consecutive generations,
            where the best-so-far solution was not improved, after which the
            switch from cfg-gp to ge will happen
        :param str name: name of the process (see
            :class:`multiprocessing.Process`)
        :keyword generator: (keyword argument) a random number generator; if
            ``None`` or not present calls to the methods of standard python
            module :mod:`random` will be performed instead
        :type generator: :class:`random.Random` , or ``None``
        :keyword int elites_num: (keyword argument) the number of best
            individuals to be copied directly to the next generation; if it
            is lower then 0 it is set to 0; default value is 0
        :keyword int tournament_size: (keyword argument) the size of
            tournament for tournament selection; if it is lower than 2 it is
            set to 2; default value is 2
        :keyword crossover_prob: (keyword argument) probability of performing a
            crossover; if it does not fit into interval [0, 1] it is set to 0 if
            lower than 0 and to 1 if higher than 1; default value is 0.8
        :keyword mutation_prob: (keyword argument) probability of performing
            a mutation; if it does not fit into interval [0, 1] it is set to
            0 if lower than 0 and to 1 if higher than 1; default value is 0.1
        :keyword prune_prob: (keyword argument) probability of performing a
            pruning; if it does not fit into interval [0, 1] it is set to 0 if
            lower than 0 and to 1 if higher than 1; default value is 0.2
        :keyword duplicate_prob: (keyword argument) probability of performing a
            duplication; if it does not fit into interval [0, 1] it is set to 0
            if lower than 0 and to 1 if higher than 1; default value is 0.2
        :param stats: stats saving class
        :type stats: :class:`wopt.evo.support.Stats`
        :param callback: a callable which will be called at the end of every
            generation with a single argument which is the algorithm instance
            itself (i.e. instance of this class)
        """
        super().__init__(name=name)

        # Positional args
        self.fitness = fitness
        self.pop_size = pop_size
        self.population_initializer = population_initializer
        self.grammar = grammar
        self.mode = mode
        if mode not in ['generational', 'steady-state']:
            raise ValueError('Argument mode must be one of \'generational\' '
                             'or \'steady-state\'')

        if isinstance(stop, int):
            # noinspection PyProtectedMember
            self.cfggp_stop = wopt.evo.cfggp.Cfggp._GenerationsStop(stop)
            # noinspection PyProtectedMember
            self.ge_stop = wopt.evo.ge.Ge._GenerationsStop(stop)
        elif callable(stop):
            self.cfggp_stop = stop
            self.ge_stop = stop
        else:
            raise TypeError('Argument stop is neither integer nor callable.')

        if stagnation_trigger <= 0:
            raise ValueError('Argument stagnation_trigger is <= 0.')
        # noinspection PyProtectedMember
        self.stagnation_stop = CfggpGe._StagnationDetector(stagnation_trigger,
                                                           self.cfggp_stop)

        # Keyword args
        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.elites_num = 0
        if 'elites_num' in kwargs:
            if not isinstance(kwargs['elites_num'], int):
                raise ValueError('Number of elites must be an integer.')
            self.elites_num = kwargs['elites_num']
            self.elites_num = max(0, self.elites_num)

        self.tournament_size = 2
        if 'tournament_size' in kwargs:
            if not isinstance(kwargs['tournament_size'], int):
                raise ValueError('Tournament size must be an integer.')
            self.tournament_size = kwargs['tournament_size']
            self.tournament_size = max(2, self.tournament_size)

        self.crossover_prob = 0.8
        if 'crossover_prob' in kwargs:
            self.crossover_prob = kwargs['crossover_prob']
            self.crossover_prob = max(0, self.crossover_prob)
            self.crossover_prob = min(1, self.crossover_prob)

        self.mutation_prob = 0.1
        if 'mutation_prob' in kwargs:
            self.mutation_prob = kwargs['mutation_prob']
            self.mutation_prob = max(0, self.mutation_prob)
            self.mutation_prob = min(1, self.mutation_prob)

        self.prune_prob = 0.2
        if 'prune_prob' in kwargs:
            self.prune_prob = kwargs['prune_prob']
            self.prune_prob = max(0, self.prune_prob)
            self.prune_prob = min(1, self.prune_prob)

        self.duplicate_prob = 0.2
        if 'duplicate_prob' in kwargs:
            self.duplicate_prob = kwargs['duplicate_prob']
            self.duplicate_prob = max(0, self.duplicate_prob)
            self.duplicate_prob = min(1, self.duplicate_prob)

        self.stats = None
        if 'stats' in kwargs:
            self.stats = kwargs['stats']

        self.callback = None
        if 'callback' in kwargs:
            self.callback = kwargs['callback']
            if not callable(self.callback):
                raise TypeError('Keyword argument callback is not a callable.')

        # Init
        self.population = []
        self.bsf = None
        self.iterations = 0

        # noinspection PyProtectedMember
        self.cfggp_initializer = CfggpGe._StaticInitializer()
        self.cfggp = wopt.evo.cfggp.Cfggp(self.fitness, self.pop_size,
                                          self.cfggp_initializer,
                                          self.grammar, self.mode,
                                          self.stagnation_stop,
                                          name='{0}(cfg)'.format(self.name),
                                          generator=self.generator,
                                          elites_num=self.elites_num,
                                          tournament_size=self.tournament_size,
                                          crossover_prob=self.crossover_prob,
                                          mutation_prob=self.mutation_prob,
                                          stats=self.stats,
                                          callback=self.callback)
        self.transition_initializer = \
            DerivationToCodonsPopulationInitializer(self.grammar)
        self.ge = wopt.evo.ge.Ge(self.fitness,
                                 self.pop_size, self.transition_initializer,
                                 self.grammar, self.mode, self.ge_stop,
                                 name='{0}(ge)'.format(self.name),
                                 generator=self.generator,
                                 elites_num=self.elites_num,
                                 tournament_size=self.tournament_size,
                                 crossover_prob=self.crossover_prob,
                                 mutation_prob=self.mutation_prob,
                                 prune_prob=self.prune_prob,
                                 duplicate_prob=self.duplicate_prob,
                                 stats=self.stats,
                                 callback=self.callback)

    def run(self):
        """Runs the CFG-GP-GE algorithm.
        """
        self.population = self.population_initializer.initialize(self.pop_size)
        self.cfggp_initializer.set_population(self.population)

        self.cfggp.run()
        self.transition_initializer.derivation_tree_population = \
            self.cfggp.population

        if self.cfggp_stop(self.cfggp):
            return
        self.stats.save_message(self.cfggp.iterations,
                                'SWITCH')

        self.ge.iterations = self.cfggp.iterations
        self.ge.bsf = self.cfggp.bsf
        self.ge.run()
