# -*- coding: utf8 -*-
"""This package contains an implementation of the Grammatical Evolution
[ONeil2003].

.. [ONeil2003] O'Neil, Michael, and Conor Ryan. `Grammatical evolution.`
    Grammatical Evolution. Springer US, 2003. 33-47.
"""
import fractions
import functools
import multiprocessing.context
import random
import gc
import copy

import wopt.evo
import wopt.evo.support
import wopt.utils


class CodonGenotypeIndividual(wopt.evo.Individual):
    """A class representing an individual as a linear string of integers.
    """

    def __init__(self, genotype, max_codon_value):
        """Creates the individual.

        :param genotype: the genotype of the individual
        :type genotype: :class:`list` of :class:`int`\\ s
        :param int max_codon_value: the maximum value of a codon in the
            genotype (exclusive, i.e. a codon will have this value minus one or
            lower)
        """
        wopt.evo.Individual.__init__(self)
        self.genotype = genotype
        self.max_codon_value = max_codon_value
        self.first_not_used = 0

    def __str__(self):
        try:
            return '{0} |=> {1}'.format(str(self.genotype), self.phenotype_str)
        except AttributeError:
            return '{0}'.format(str(self.genotype))

    def copy(self, carry_evaluation=True):
        clone = CodonGenotypeIndividual(list(self.genotype),
                                        self.max_codon_value)
        if carry_evaluation:
            clone.fitness = copy.deepcopy(self.fitness)
            clone.first_not_used = self.first_not_used
        return clone

    def set_first_not_used(self, first_not_used):
        self.first_not_used = first_not_used

    def get_first_not_used(self):
        return self.first_not_used

    def get_max_codon_value(self):
        return self.max_codon_value

    def get_codon(self, index):
        return self.genotype[index]

    def set_codon(self, index, new_codon):
        if not (0 <= new_codon < self.max_codon_value):
            raise ValueError(('Codon value must be in range [0, {0}) but was'
                              ' {1}.').format(self.max_codon_value, new_codon))
        self.genotype[index] = new_codon

    def get_codon_num(self):
        return len(self.genotype)


class RandomCodonGenotypeInitializer(wopt.evo.IndividualInitializer):
    """Generates a genotype with a random length within a given range and
    random codons.
    """

    def __init__(self, min_length, max_length, **kwargs):
        """Creates an initializer with given parameters.

        The genotypes generated by the
        :meth:`RandomCodonGenotypeInitializer.initialize` are going to have
        the minimum length of ``min_length`` and maximum length of
        ``max_length``.

        The optional keyword argument ``max_codon_value`` controls the maximum
        integer value of each codon in the generated genotypes.

        The optional keyword argument ``generator`` can be used to pass a
        random number generator to the initializer which is to be used for
        generation. If it is ``None`` or not present a standard generator is
        used which is the :mod:`random` module and its functions. If a
        generator is passed it is expected to have the corresponding methods
        to the :mod:`random` module (individual.e. the class
        :mod:`random`\\ .Random).

        .. warning::

            If default generator is used (individual.e. the methods of
            :mod:`random`) it is assumed that it is already seeded and no seed
            is set inside this class.

        :param int min_length: minimum length of the genotype
        :param int max_length: maximum length of the genotype
        :keyword int max_codon_value: (keyword argument) maximum value a codon
            can have

            if ``None`` or not present default value of 255 is used
        :keyword generator: a random number generator; if ``None`` or not
            present calls to the methods of standard python module
            :mod:`random` will be performed instead
        :type generator: :mod:`random`\\ .Random or ``None``
        :return: a randomly generated individual
        :rtype: :class:`CodonGenotypeIndividual`
        """
        wopt.evo.IndividualInitializer.__init__(self)

        self.min_length = min_length
        self.max_length = max_length

        if 'max_codon_value' in kwargs:
            self.max_codon_value = kwargs['max_codon_value']
        else:
            self.max_codon_value = 255

        if 'generator' in kwargs:
            self.generator = kwargs['generator']
        else:
            self.generator = random

    def initialize(self):
        genotype = []
        for _ in range(self.generator.randint(self.min_length,
                                              self.max_length)):
            genotype.append(self.generator.randint(0, self.max_codon_value))
        return CodonGenotypeIndividual(genotype, self.max_codon_value)


class RandomWalkInitializer(wopt.evo.IndividualInitializer):
    """Generates a codon genotype by random walk through a grammar (i.e. the
    resulting genotypes encode exactly a complete derivation tree).
    """

    def __init__(self, grammar, **kwargs):
        """Creates an initializer with given parameters.

        The optional keyword argument ``generator`` can be used to pass a
        random number generator to the initializer which is to be used for
        generation. If it is ``None`` or not present a standard generator is
        used which is the :mod:`random` module and its functions. If a
        generator is passed it is expected to have the corresponding methods
        to the :mod:`random` module (individual.e. the class
        :mod:`random`\\ .Random).

        .. warning::

            Whatever generator is used it is assumed that it is already seeded
            and no seed is set inside this class.

        :param grammar: the grammar to generate
        :type grammar: :class:`wopt.evo.support.grammar.Grammar`
        :keyword generator: a random number generator; if ``None`` or not
            present calls to the methods of standard python module
            :mod:`random` will be performed instead
        :type generator: :mod:`random`\\ .Random or ``None``
        :keyword max_depth: maximum depth of the corresponding derivation tree;
            if ``None`` or not present default value of infinity is used
        :keyword multiplier: number which will be used to multiply the LCM of
            all choices numbers to get a higher maximum codon value (default
            is 1, i.e. maximum codon value will be a LCM of numbers of all
            choices in the grammar)
        :return: a randomly generated individual
        :rtype: :class:`DerivationTreeIndividual`
        """
        wopt.evo.IndividualInitializer.__init__(self)

        self.grammar = grammar

        if 'generator' in kwargs:
            self.generator = kwargs['generator']
        else:
            self.generator = random

        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']
        else:
            self.max_depth = float('inf')

        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']
        else:
            self.multiplier = 1

        choice_nums = [r.get_choices_num() for r in grammar.get_rules()]
        m = functools.reduce(lambda a, b: a * b // fractions.gcd(a, b),
                             choice_nums)
        self.max_choices = m * self.multiplier

    def initialize(self):
        iterator = wopt.utils.RandomIntIterable(-1, -1,
                                                0, self.max_choices - 1,
                                                generator=self.generator)
        sequence = []
        _ = self.grammar.to_tree(decisions=iterator,
                                 max_wraps=0,
                                 max_depth=self.max_depth,
                                 sequence=sequence)
        return CodonGenotypeIndividual(sequence,
                                       self.max_choices)


class Ge(multiprocessing.context.Process):
    """This class forms the whole GE algorithm.

    This is a basic form of an algorithm using a generational scheme, elitism,
    tournament selection, single-point crossover and codon-level mutation.

    Derive from this class to implement custom algorithm structure.
    """

    class _GenerationsStop(object):

        def __init__(self, generations):
            self.generations = generations

        def __call__(self, ge):
            if ge.mode == 'generational':
                return ge.iterations >= self.generations
            if ge.mode == 'steady-state':
                return ge.iterations >= (self.generations * ge.pop_size)

    def __init__(self, fitness, pop_size, population_initializer, grammar, mode,
                 stop, name=None, **kwargs):
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
            self.stop = Ge._GenerationsStop(stop)
        elif callable(stop):
            self.stop = stop
        else:
            raise TypeError('Argument stop is neither integer nor callable.')

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

        self.population = []
        self.bsf = None
        self.iterations = 0

    def run(self):
        """Runs the GE algorithm.
        """
        try:
            self.population = self.population_initializer.initialize(
                self.pop_size)

            if self.mode == 'generational':
                self._run_generational()
            elif self.mode == 'steady-state':
                self._run_steady_state()
        finally:
            gc.collect()
            try:
                self.stats.cleanup()
            except AttributeError:
                pass

    def _run_generational(self):
        while not self.stop(self):
            if self.callback is not None:
                self.callback(self)
            elites = self.extract_elites()

            others = []
            while True:
                o1 = self.select_tournament(self.population).copy()
                o2 = self.select_tournament(self.population).copy()

                self.prune(o1)
                self.prune(o2)

                self.crossover(o1, o2)

                self.mutate(o1)
                self.mutate(o2)

                self.duplicate(o1)
                self.duplicate(o2)

                if len(self.population) - (len(others) + len(elites)) >= 2:
                    others.append(o1)
                    others.append(o2)
                elif len(self.population) - (len(others) + len(elites)) >= 1:
                    if self.generator.random() < 0.5:
                        o1 = o2
                    others.append(o1)
                else:
                    break
            self.population = elites + others
            self.iterations += 1

    def _run_steady_state(self):
        for i in self.population:
            self.evaluate(i)

        self.fitness.sort(self.population)

        while not self.stop(self):
            if self.callback is not None:
                self.callback(self)
            o1 = self.select_tournament(self.population).copy()
            o2 = self.select_tournament(self.population).copy()

            self.prune(o1)
            self.prune(o2)

            self.crossover(o1, o2)

            self.mutate(o1)
            self.mutate(o2)

            self.duplicate(o1)
            self.duplicate(o2)

            self.steady_state_replace(o1, o2)
            self.iterations += 1

    def evaluate(self, individual):
        self.fitness.evaluate(individual)
        if self.bsf is None or self.fitness.compare(individual.get_fitness(),
                                                    self.bsf.get_fitness()):
            self.bsf = individual
            self.stats.save_bsf(self.iterations, self.bsf)

    def extract_elites(self):
        if self.elites_num == 0:
            return []

        for individual in self.population:
            if individual.get_fitness() is None:
                self.evaluate(individual)

        # self.generator.shuffle(self.population)
        self.population.sort(key=lambda x: x.get_fitness(),
                             reverse=self.fitness.maximize())
        return self.population[0:self.elites_num]

    def select_tournament(self, population):
        candidates = self.generator.sample(population, self.tournament_size)
        best = None
        for candidate in candidates:
            if candidate.get_fitness() is None:
                self.evaluate(candidate)

            if best is None or self.fitness.compare(candidate.get_fitness(),
                                                    best.get_fitness()):
                best = candidate
        return best

    def crossover(self, o1, o2):
        if self.generator.random() < self.crossover_prob:
            self.single_point_crossover(o1, o2)

    def single_point_crossover(self, o1, o2):
        if not isinstance(o1, CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        if not isinstance(o2, CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')

        g1 = o1.genotype
        g2 = o2.genotype

        if len(g1) == len(g2) == 1:
            return

        assert g1, g1
        assert g2, g2

        if len(g1) == 1:
            point1 = self.generator.randint(0, 1)
        else:
            point1 = self.generator.randrange(1, len(g1))
        if len(g2) == 1:
            point2 = self.generator.randint(0, 1)
        else:
            point2 = self.generator.randrange(1, len(g2))

        o1.genotype = g1[:point1] + g2[point2:]
        o2.genotype = g2[:point2] + g1[point1:]

        assert o1.genotype, (o1.genotype, g1, g2, point1, point2)
        assert o2.genotype, (o2.genotype, g1, g2, point1, point2)

        o1.set_fitness(None)
        o2.set_fitness(None)

    def mutate(self, individual):
        self.codon_change_mutate(individual)

    def codon_change_mutate(self, individual):
        if not isinstance(individual, CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        mutated = False
        for i in range(individual.get_codon_num()):
            if self.generator.random() < self.mutation_prob:
                new_codon = self.generator.randrange(individual.
                                                     get_max_codon_value())
                individual.set_codon(i, new_codon)
                mutated = True
        if mutated:
            individual.set_fitness(None)

    def prune(self, individual):
        if not isinstance(individual, CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        if (individual.get_first_not_used() < individual.get_codon_num() and
                self.generator.random() < self.prune_prob):
            individual.genotype = individual.genotype[:individual.
                                                      get_first_not_used()]

    def duplicate(self, individual):
        if not isinstance(individual, CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        if self.generator.random() < self.duplicate_prob:
            pos = self.generator.randrange(individual.get_codon_num())
            n = self.generator.randint(1, individual.get_codon_num() - pos)

            individual.genotype = (individual.genotype[:-1] +
                                   individual.genotype[pos:pos + n] +
                                   individual.genotype[-1:])
            individual.set_fitness(None)

    def steady_state_replace(self, o1, o2):
        if o1.get_fitness() is None:
            self.evaluate(o1)
        if o2.get_fitness() is None:
            self.evaluate(o2)

        if self.fitness.compare(o1.get_fitness(), o2.get_fitness()):
            o = o1
        elif self.fitness.compare(o2.get_fitness(), o1.get_fitness()):
            o = o2
        else:
            if self.generator.random() < 0.5:
                o = o1
            else:
                o = o2

        if self.fitness.compare(self.population[-1].get_fitness(),
                                o.get_fitness()):
            return

        self.population.pop(-1)

        if self.fitness.compare(self.population[-1].get_fitness(),
                                o.get_fitness()):
            self.population.append(o)
            return

        if self.fitness.compare(o.get_fitness(),
                                self.population[0].get_fitness()):
            self.population.insert(0, o)
            return

        l = 0
        u = len(self.population)
        c = (l + u) // 2
        while l < u and l != c != u:
            ci = self.population[c]
            if self.fitness.compare(ci.get_fitness(), o.get_fitness()):
                l = c
            elif self.fitness.compare(o.get_fitness(), ci.get_fitness()):
                u = c
            else:
                break
            c = (l + u) // 2
        self.population.insert(c + 1, o)