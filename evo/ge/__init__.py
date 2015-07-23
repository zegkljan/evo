# -*- coding: utf8 -*-
"""This package contains an implementation of the Grammatical Evolution
[ONeil2003].

.. [ONeil2003] O'Neil, Michael, and Conor Ryan. `Grammatical evolution.`
    Grammatical Evolution. Springer US, 2003. 33-47.
"""

import functools
import multiprocessing.context
import random
import gc
import math
import logging
import pprint

import evo
import evo.utils
import evo.utils.grammar
import evo.utils.random
import evo.ge.support

__author__ = 'Jan Žegklitz'


class Ge(evo.GeneticBase, multiprocessing.context.Process):
    """This class forms the whole GE algorithm.
    """

    LOG = logging.getLogger(__name__ + '.Ge')

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

        :param evo.Fitness fitness: fitness used to evaluate individual
            performance
        :param int pop_size: size of the population; this value will be
            passed to the ``population_initializer``\ 's method
            ``initialize``\ ()
        :param ge.init.PopulationInitializer population_initializer: initializer
            used to initialize the initial population
        :param evo.support.grammar.Grammar grammar: grammar this algorithm
            operates on
        :param mode: Specifies which mode of genetic algorithm to use. Possible
            values are ``'generational'`` and ``'steady-state'``\ .
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
        :keyword int elites_num: (keyword argument) the number of best
            individuals to be copied directly to the next generation; if it
            is lower then 0 it is set to 0; default value is 0
        :keyword int tournament_size: (keyword argument) the size of
            tournament for tournament selection; if it is lower than 2 it is
            set to 2; default value is 2
        :keyword crossover_prob: (keyword argument) probability of performing a
            crossover; if it does not fit into interval [0, 1] it is set to 0 if
            lower than 0 and to 1 if higher than 1; default value is 0.8
        :keyword crossover_type: (keyword argument) the type of crossover;
            possible values are

                * ``'ripple'`` - the ripple crossover
                * ``'subtree'`` - the subtree crossover
                * ``('variable', pref_change_prob, (method1, method2, ...))`` -
                  the variable crossover; ``pref_change_prob`` is then the
                  probability of changing the preferred crossover method in the
                  children, ``method1, ...`` are the particular methods which
                  can be one of the previous ones (in the same form)

            The default value is ``'ripple'``\ .
        :keyword mutation_prob: (keyword argument) probability of performing
            a mutation; if it does not fit into interval [0, 1] it is set to
            0 if lower than 0 and to 1 if higher than 1; default value is 0.1
        :keyword mutation_type: (keyword argument) the type of mutation;
            possible values are

                * ``'codon-change'`` - codon-chagne mutation
                * ``('subtree', max_depth)`` - derivation subtree mutation;
                  ``max_depth`` is the maximum depth of the randomly generated
                  subtree
                * ``('variable', pref_change_prob, (method1, method2, ...))`` -
                  a variable mutation; ``pref_change_prob`` is then the
                  probability of changing the preferred mutation method in the
                  mutated individual, ``method1, ...`` are the particular
                  methods which can be one of the previous ones (in the same
                  form)

            The default value is ``'codon-change'``\ .
        :keyword prune_prob: (keyword argument) probability of performing a
            pruning; if it does not fit into interval [0, 1] it is set to 0 if
            lower than 0 and to 1 if higher than 1; default value is 0.2
        :keyword duplicate_prob: (keyword argument) probability of performing a
            duplication; if it does not fit into interval [0, 1] it is set to 0
            if lower than 0 and to 1 if higher than 1; default value is 0.2
        :keyword evo.support.Stats stats: stats saving class
        :keyword callback: a callable which will be called at the beginning of
            every generation with a single argument which is the algorithm
            instance itself (i.e. instance of this class)
        """
        evo.GeneticBase.__init__(self)
        multiprocessing.context.Process.__init__(self, name=name)

        # Positional args
        self.fitness = fitness
        self.pop_size = pop_size
        self.population_initializer = population_initializer
        if isinstance(grammar, evo.utils.grammar.Grammar):
            self.grammar = grammar
        else:
            self.grammar = evo.utils.grammar.Grammar(grammar)
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

        self.crossover_method = self.single_point_crossover
        self.crossover_method_args = ()
        if 'crossover_type' in kwargs and kwargs['crossover_type'] is not None:
            self.crossover_method, self.crossover_method_args = \
                self.setup_crossover(kwargs['crossover_type'])

        self.mutation_prob = 0.1
        if 'mutation_prob' in kwargs:
            self.mutation_prob = kwargs['mutation_prob']
            self.mutation_prob = max(0, self.mutation_prob)
            self.mutation_prob = min(1, self.mutation_prob)

        self.mutate_method = self.codon_change_mutate
        self.mutate_method_args = ()
        if 'mutation_type' in kwargs and kwargs['mutation_type'] is not None:
            self.mutate_method, self.mutate_method_args = \
                self.setup_mutation(kwargs['mutation_type'])

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

        self.replace_method = self.random_replace
        self.replace_method_args = ()
        if 'steady_state_replace' in kwargs:
            ssr = kwargs['steady_state_replace']
            if ssr == 'random':
                self.replace_method = self.random_replace
                self.replace_method_args = ()
            elif ssr == 'inverse-tournament':
                self.replace_method = self.inverse_tournament_replace
                self.replace_method_args = (self.tournament_size, False)
            elif ssr[0] == 'inverse-tournament':
                try:
                    self.replace_method = self.inverse_tournament_replace
                    self.replace_method_args = (int(ssr[1]), bool(ssr[2]))
                except:
                    raise ValueError('Invalid parameters for inverse '
                                     'tournament replacement strategy.')
            else:
                raise ValueError('Invalid steady state replacement strategy.')

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
        self.bsf = None

        self.iterations = 0
        """
        The number of elapsed iterations of the algorithm (either generations
        in the generational mode or just iterations in the steady-state mode).
        """

    def setup_crossover(self, crossover_type):
        """Helper method for the constructor which sets up the crossover method.
        """
        if crossover_type == 'ripple':
            crossover_method = self.single_point_crossover
            crossover_method_args = ()
        elif crossover_type == 'subtree':
            crossover_method = self.subtree_crossover
            crossover_method_args = ()
        elif crossover_type[0] == 'variable':
            crossover_method = self.variable_crossover
            scs = []
            for subcrossover in crossover_type[2]:
                cm, cma = self.setup_crossover(subcrossover)
                scs.append((cm, cma))
            crossover_method_args = (crossover_type[1], tuple(scs))
        else:
            raise ValueError('Invalid crossover type.')
        return crossover_method, crossover_method_args

    def setup_mutation(self, mutation_type):
        """Helper method for the constructor which sets up the mutation method.
        """
        if mutation_type == 'codon-change':
            mutation_method = self.codon_change_mutate
            mutation_method_args = ()
        elif mutation_type[0] == 'subtree':
            mutation_method = self.subtree_mutate
            mutation_method_args = (mutation_type[1],)
        elif mutation_type[0] == 'variable':
            mutation_method = self.variable_mutation
            sms = []
            for submutation in mutation_type[2]:
                mm, mma = self.setup_mutation(submutation)
                sms.append((mm, mma))
            mutation_method_args = (mutation_type[1], tuple(sms))
        else:
            raise ValueError('Invalid crossover type.')
        return mutation_method, mutation_method_args

    def run(self):
        """Runs the GE algorithm.
        """
        Ge.LOG.info('Starting algorithm.')
        try:
            self.population = self.population_initializer.initialize(
                self.pop_size)

            if self.mode == 'generational':
                self._run_generational()
            elif self.mode == 'steady-state':
                self._run_steady_state()
        finally:
            Ge.LOG.info('Finished.\nFitness: %f\n%s', self.bsf.get_fitness(),
                        pprint.pformat(self.bsf.get_data()))
            Ge.LOG.info('Performing garbage collection.')
            gc.collect()
            try:
                if self.stats is not None:
                    self.stats.cleanup()
            except AttributeError:
                pass

    def _run_generational(self):
        Ge.LOG.info('Starting generational evolution.')
        while not self.stop(self):
            Ge.LOG.info('Starting iteration %d', self.iterations)
            if self.callback is not None:
                self.callback(self)
            elites = self.extract_elites()

            Ge.LOG.debug('Processing selection.')
            others = []
            while True:
                o1 = self.select_tournament(self.population,
                                            self.tournament_size,
                                            sorted_=self.population_sorted). \
                    copy()
                o2 = self.select_tournament(self.population,
                                            self.tournament_size,
                                            sorted_=self.population_sorted). \
                    copy()

                self.prune(o1)
                self.prune(o2)

                offsprings = self.crossover(o1, o2)

                # noinspection PyTypeChecker
                for o in offsprings:
                    self.mutate(o)
                    self.duplicate(o)
                    if self.pop_size - (len(others) + len(elites)) > 0:
                        self.test_bsf(o)
                        others.append(o)
                    else:
                        break
                if self.pop_size - (len(others) + len(elites)) <= 0:
                    break
            self.population = elites + others
            Ge.LOG.info('Finished iteration %d', self.iterations)
            self.iterations += 1
        if self.callback is not None:
            self.callback(self)
        Ge.LOG.info('Finished generational evolution.')

    def _run_steady_state(self):
        self.population_sorted = self.fitness.sort(self.population, False,
                                                   evo.Fitness.
                                                   COMPARE_TOURNAMENT)

        while not self.stop(self):
            if self.callback is not None:
                self.callback(self)
            o1 = self.select_tournament(self.population,
                                        self.tournament_size,
                                        sorted_=self.population_sorted).copy()
            o2 = self.select_tournament(self.population,
                                        self.tournament_size,
                                        sorted_=self.population_sorted).copy()

            self.prune(o1)
            self.prune(o2)

            offsprings = self.crossover(o1, o2)

            # noinspection PyTypeChecker
            for o in offsprings:
                self.mutate(o)
                self.duplicate(o)
                self.replace(o)
                self.test_bsf(o)
            self.iterations += 1
        if self.callback is not None:
            self.callback(self)

    def test_bsf(self, individual):
        self.fitness.evaluate(individual)
        if self.bsf is None or self.fitness.is_better(individual,
                                                      self.bsf,
                                                      evo.Fitness.COMPARE_BSF):
            self.bsf = individual
            if self.stats is not None:
                self.stats.save_bsf(self.iterations, self.bsf)

    def extract_elites(self):
        Ge.LOG.debug('Extracting %d elites.', self.elites_num)
        if self.elites_num == 0:
            return []

        self.population_sorted = self.fitness.sort(self.population, False,
                                                   evo.Fitness.
                                                   COMPARE_TOURNAMENT)
        if self.population_sorted:
            self.test_bsf(self.population[0])
            return self.population[0:self.elites_num]

        Ge.LOG.debug('Population not sorted by fitness, using explicit '
                     'sorting.')
        cmp = lambda a, b: self.fitness.is_better(a, b, evo.Fitness.
                                                  COMPARE_TOURNAMENT)
        sorted_population = sorted(self.population,
                                   key=functools.cmp_to_key(cmp))
        self.test_bsf(sorted_population[0])
        Ge.LOG.debug('Elites extracted.')
        return sorted_population[0:self.elites_num]

    def select_tournament(self, population, size, inverse=False, sorted_=True):
        return population[self.select_tournament_idx(population, size,
                                                     inverse, sorted_)]

    def select_tournament_idx(self, population, size,
                              inverse=False, sorted_=False):
        candidates_idx = self.generator.sample(range(len(population)), size)
        if sorted_:
            if inverse:
                return max(candidates_idx)
            return min(candidates_idx)
        best_idx = None
        for candidate_idx in candidates_idx:
            candidate = population[candidate_idx]

            if best_idx is None:
                best_idx = candidate_idx
            else:
                better = self.fitness.is_better(candidate, population[best_idx],
                                                evo.Fitness.
                                                COMPARE_TOURNAMENT)
                if bool(inverse) ^ bool(better):
                    best_idx = candidate_idx
        return best_idx

    def crossover(self, o1, o2):
        """Performs a crossover of two individuals.

        :param evo.ge.support.CodonGenotypeIndividual o1: first parent
        :param evo.ge.support.CodonGenotypeIndividual o2: second parent
        """
        if self.generator.random() >= self.crossover_prob:
            return [o1, o2]
        Ge.LOG.debug('Performing crossover of individuals %s, %s', o1, o2)
        assert self.crossover_method is not None
        # noinspection PyArgumentList
        return self.crossover_method(o1, o2, *self.crossover_method_args)

    def variable_crossover(self, o1, o2, crossover_pref_change_prob,
                           crossover_methods):
        Ge.LOG.debug('Variable crossover of individuals %s, %s', o1, o2)
        xover_pref = o1.get_data('xover-pref')
        if xover_pref is None:
            xover_pref = self.generator.randrange(len(crossover_methods))

        xover_method, args = crossover_methods[xover_pref]
        offsprings = xover_method(o1, o2, *args)
        for o in offsprings:
            if self.generator.random() < crossover_pref_change_prob:
                o.set_data('xover-pref',
                           self.generator.randrange(len(crossover_methods)))
            else:
                o.set_data('xover-pref', xover_pref)
        return offsprings

    # noinspection PyUnusedLocal
    def single_point_crossover(self, o1, o2, *args):
        if not isinstance(o1, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        if not isinstance(o2, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        Ge.LOG.debug('Ripple crossover of individuals %s, %s', o1, o2)

        g1 = o1.genotype
        g2 = o2.genotype

        if len(g1) == len(g2) == 1:
            return [o1, o2]

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

        o1.set_annotations(None)
        o2.set_annotations(None)

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]

    # noinspection PyUnusedLocal
    def subtree_crossover(self, o1, o2, *args):
        """Performs the subtree crossover on the two given parents.

        First, a random point in ``o1`` is chosen. Then the codons in ``o2`` are
        filtered to those with the same non-terminal label. If there are none a
        new random point in ``o1`` is chosen. If there are some one of them is
        chosen randomly. The subsequences starting at the chosen points and of
        lengths defined by the annotations are swapped.

        If one of the individuals does not have any annotations (it was too
        short to fully expand) the method returns an empty list (i.e. the
        crossover failed - no offsprings were generated).
        """
        if not isinstance(o1, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        if not isinstance(o2, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        Ge.LOG.debug('Subtree crossover of individuals %s, %s', o1, o2)

        if not o1.get_annotations() or not o2.get_annotations():
            return []

        g1 = o1.genotype
        g2 = o2.genotype

        a1 = list(enumerate(o1.get_annotations()))
        a2 = list(enumerate(o2.get_annotations()))
        a1 = list(filter(lambda x: x[1] is not None, a1))
        a2 = list(filter(lambda x: x[1] is not None, a2))

        if len(g1) == len(g2) == 1:
            return [o1, o2]

        assert g1, g1
        assert g2, g2
        assert a1, a1
        assert a2, a2

        point1 = self.generator.randrange(len(a1))
        _, (rule, l1) = a1[point1]
        while True:
            a2ok = [x for x in a2 if x[1][0] == rule]
            if a2ok:
                break
            else:
                point1 = self.generator.randrange(len(a1))
                _, (rule, l1) = a1[point1]
        # noinspection PyUnboundLocalVariable
        point2 = self.generator.randrange(len(a2ok))
        point2, (_, l2) = a2ok[point2]

        o1.genotype = g1[:point1] + g2[point2:point2 + l2] + g1[point1 + l1:]
        o2.genotype = g2[:point2] + g1[point1:point1 + l1] + g2[point2 + l2:]

        a1 = o1.get_annotations()
        a2 = o2.get_annotations()
        new_a1 = a1[:point1] + a2[point2:point2 + l2] + a1[point1 + l1:]
        new_a2 = a2[:point2] + a1[point1:point1 + l1] + a2[point2 + l2:]
        o1.set_annotations(new_a1)
        o2.set_annotations(new_a2)
        # o1.set_annotations(None)
        # o2.set_annotations(None)

        assert o1.genotype, (o1.genotype, g1, g2, point1, point2)
        assert o2.genotype, (o2.genotype, g1, g2, point1, point2)

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]

    def mutate(self, individual):
        self.mutate_method(individual, *self.mutate_method_args)

    def variable_mutation(self, individual, mutation_pref_change_prob,
                          mutation_methods):
        Ge.LOG.debug('Variable mutation of individual %s', individual)
        mutation_pref = individual.get_data('mutation-pref')
        if mutation_pref is None:
            mutation_pref = self.generator.randrange(len(mutation_methods))

        mutation_method, args = mutation_methods[mutation_pref]
        out = mutation_method(individual, *args)
        while out is None:
            mutation_pref = (mutation_pref + 1) % len(mutation_methods)
            out = mutation_method(individual, *args)

        if self.generator.random() < mutation_pref_change_prob:
            individual.set_data('mutation-pref',
                                self.generator.randrange(len(mutation_methods)))
        else:
            individual.set_data('mutation-pref', mutation_pref)
        return out

    def codon_change_mutate(self, individual):
        if not isinstance(individual, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        if self.mutation_prob == 0:
            return individual
        if self.mutation_prob == 1:
            Ge.LOG.debug('Mutating individual %s', individual.__str__())
            for i in range(individual.get_codon_num()):
                new_codon = self.generator.randrange(individual.
                                                     get_max_codon_value())
                individual.set_codon(i, new_codon)
            individual.set_fitness(None)
            return individual

        mutated = False
        k = math.floor(math.log10(1 - self.generator.random()) /
                       math.log10(1 - self.mutation_prob))
        while k < individual.get_codon_num():
            if not mutated:
                Ge.LOG.debug('Mutating individual %s', individual.__str__())
            new_codon = self.generator.randrange(individual.
                                                 get_max_codon_value())
            individual.set_codon(k, new_codon)
            mutated = True
            k += math.floor(math.log10(1 - self.generator.random()) /
                            math.log10(1 - self.mutation_prob))
        if mutated:
            individual.set_fitness(None)
        return individual

    def subtree_mutate(self, individual, max_depth):
        if not isinstance(individual, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        if self.mutation_prob == 0:
            return individual
        if self.generator.random() < self.mutation_prob:
            Ge.LOG.debug('Mutating individual %s', individual.__str__())
            if individual.annotations is None:
                Ge.LOG.warn('No annotations to mutate %s',
                             individual.__str__())
                return None
            annotations = individual.get_annotations()
            idx = self.generator.randrange(len(annotations))
            (new_codons, new_annotations) = self.generate_subtree_codons(
                annotations[idx][0], max_depth)
            new_genotype = (individual.genotype[0:idx] + new_codons +
                            individual.genotype[idx + annotations[idx][1]:])
            new_annotations = (annotations[0:idx] + new_annotations +
                               annotations[idx + annotations[idx][1]:])
            individual.set_annotations(new_annotations)
            individual.genotype = new_genotype
            individual.set_fitness(None)
        return individual

    def generate_subtree_codons(self, start_symbol, max_depth):
        ri = evo.utils.random.RandomIntIterable(
            -1, -1, 0, self.grammar.get_choice_nums_lcm(),
            generator=self.generator)
        seq = []
        o, _, _, _, annotations = self.grammar.to_tree(ri, 0,
                                                       max_depth=max_depth,
                                                       sequence=seq,
                                                       start_rule=start_symbol)
        return seq, annotations

    def prune(self, individual):
        if not isinstance(individual, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        if (individual.get_first_not_used() < individual.get_codon_num() and
                self.generator.random() < self.prune_prob):
            Ge.LOG.debug('Pruning individual %s', individual.__str__())
            individual.genotype = individual.genotype[:individual.
                                                      get_first_not_used()]

    def duplicate(self, individual):
        if not isinstance(individual, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Individual must be of type '
                            'CodonGenotypeIndividual.')
        r = self.generator.random()
        if r < self.duplicate_prob:
            Ge.LOG.debug('Duplicating individual %s', individual.__str__())
            pos = self.generator.randrange(individual.get_codon_num())
            n = self.generator.randint(1, individual.get_codon_num() - pos)

            individual.genotype = (individual.genotype[:-1] +
                                   individual.genotype[pos:pos + n] +
                                   individual.genotype[-1:])
            individual.set_fitness(None)

    def replace(self, indiv):
        self.replace_method(indiv, *self.replace_method_args)

    def random_replace(self, indiv):
        self._pop_replace(self.generator.randrange(len(self.population)), indiv)

    def inverse_tournament_replace(self, indiv, n, participate):
        if participate:
            loser_idx = self.select_tournament_idx(self.population, n - 1, True,
                                                   sorted_=self.
                                                   population_sorted)
            loser = self.population[loser_idx]
            if self.fitness.is_better(indiv, loser,
                                      evo.Fitness.COMPARE_TOURNAMENT):
                self._pop_replace(loser_idx, indiv)
        else:
            loser_idx = self.select_tournament_idx(self.population, n, True,
                                                   sorted_=self.
                                                   population_sorted)
            self._pop_replace(loser_idx, indiv)

    def steady_state_replace(self, o1, o2):
        if o1.get_fitness() is None:
            self.test_bsf(o1)
        if o2.get_fitness() is None:
            self.test_bsf(o2)

        if self.fitness.is_better(o1, o2, evo.Fitness.COMPARE_TOURNAMENT):
            o = o1
        elif self.fitness.is_better(o2, o1, evo.Fitness.COMPARE_TOURNAMENT):
            o = o2
        else:
            if self.generator.random() < 0.5:
                o = o1
            else:
                o = o2

        if self.fitness.is_better(self.population[-1], o,
                                  evo.Fitness.COMPARE_TOURNAMENT):
            return

        self.population.pop(-1)

        if self.fitness.is_better(self.population[-1], o,
                                  evo.Fitness.COMPARE_TOURNAMENT):
            self.population.append(o)
            return

        if self.fitness.is_better(o, self.population[0],
                                  evo.Fitness.COMPARE_TOURNAMENT):
            self.population.insert(0, o)
            return

        l = 0
        u = len(self.population)
        c = (l + u) // 2
        while l < u and l != c != u:
            ci = self.population[c]
            if self.fitness.is_better(ci, o, evo.Fitness.COMPARE_TOURNAMENT):
                l = c
            elif self.fitness.is_better(o, ci,
                                        evo.Fitness.COMPARE_TOURNAMENT):
                u = c
            else:
                break
            c = (l + u) // 2
        self.population.insert(c + 1, o)


# noinspection PyAbstractClass
class GeFitness(evo.Fitness):
    """This class is a base class for fitness used with Grammatical Evolution.

    This class takes care of the machinery regarding the individual decoding
    process:

        1. The genotype is decoded using the :meth:`.decode` method.
        2. The decoding output is passed to the method :meth:`.make_phenotype`
           which turns the decoding output to a *phenotype*.
        3. The *phenotype* is passed to the method
           :meth:`.evaluate_phenotype` which returns the *fitness* of the
           phenotype. The *individual* is passed to this method too for the
           subclasses to be able to store additional data to the individual.
        4. The *fitness* and decoding annotations are assigned to the original
           individual using the :meth:`evo.Individual.set_fitness` and
           :meth:`evo.Ge.support.CodonGenotypeIndividual.set_annotations`
           methods.

    The individuals passed to the :meth:`.evaluate` method are expected to be of
    class :class:`evo.ge.support.CodonGenotypeIndividual`.

    The deriving classes are to implement the following methods:

        * :meth:`.decode`
        * :meth:`.make_phenotype`
        * :meth:`.evaluate_phenotype`

    .. seealso::

        Class :class:`evo.ge.GeTreeFitness`
            Decodes the individual to a derivation tree.
        Class :class:`evo.ge.GeTextFitness`
            Decodes the individual to text.
    """

    class NotFinishedError(Exception):
        pass

    def __init__(self, grammar, unfinished_fitness, wraps=0,
                 skip_if_evaluated=True):
        """
        :param grammar: a grammar to use for decoding
        :type grammar: either :class:`evo.utils.grammar.Grammar` or an argument
            for its constructor
        :param unfinished_fitness: a fitness value to assign to individuals
            which did were not able to decode completely (i.e. there were some
            unexpanded non-terminals left after the decoding ended)
        :param int wraps: number of wraps (i.e. reusing the codon sequence from
            beginning after the end is reached); default is 0
        :param bool skip_if_evaluated: If ``True`` (default) then the evaluation
            process (incl. decoding) will not be performed at all, if the
            individual's ``get_fitness`` method returns a non-\ ``None`` value.
            If ``False`` then the evaluation will always be carried out.
        """
        self.grammar = None
        if isinstance(grammar, evo.utils.grammar.Grammar):
            self.grammar = grammar
        else:
            self.grammar = evo.utils.grammar.Grammar(grammar)

        self.unfinished_fitness = unfinished_fitness
        self.wraps = wraps
        self.skip_if_evaluated = skip_if_evaluated

    def evaluate(self, individual):
        """
        :param evo.ge.support.CodonGenotypeIndividual individual: individual to
            decode
        """
        if self.skip_if_evaluated and individual.get_fitness() is not None:
            return

        try:
            decoded = self.decode(individual)
        except GeFitness.NotFinishedError:
            individual.set_fitness(self.unfinished_fitness)
            return

        phenotype = self.make_phenotype(decoded, individual)
        fitness = self.evaluate_phenotype(phenotype, individual)

        individual.set_fitness(fitness)
        pass

    def decode(self, individual):
        """Decodes the individual.

        This method is to be implemented. Apart from encoding, this method must

            * Raise a :class:`evo.ge.GeFitness.NotFinishedError` error if the
              decoding process could not be finished.
            * Set the decoding annotations back to the individual.

        :param evo.ge.support.CodonGenotypeIndividual individual: the individual
            to decode
        :raises evo.ge.GeFitness.NotFinishedError: the decoding process could
            not be finished
        """
        raise NotImplementedError()

    def make_phenotype(self, decoded, individual):
        """Transforms the output of the decoding process to a *phenotype*\ .

        :param decoded: the result of :meth:`.decode`
        :param individual: the base individual; may be useful to store
            information about the phenotype creation
        """
        raise NotImplementedError()

    def evaluate_phenotype(self, phenotype, individual):
        """Evaluates the phenotype.
        """
        raise NotImplementedError()


# noinspection PyAbstractClass
class GeTreeFitness(GeFitness):
    """Base class for fitnesses that use a derivation as the decoding output.

    Implement the :meth:`.parse_derivation_tree` method to transform the
    derivation tree to the phenotype.
    """
    def decode(self, individual):
        (derivation_tree,
         finished,
         used_num,
         wraps,
         annotations) = self.grammar.to_tree(individual.genotype, self.wraps)
        individual.set_annotations(annotations)
        individual.set_first_not_used(used_num)
        if not finished:
            raise GeFitness.NotFinishedError()

        return derivation_tree

    def make_phenotype(self, decoded, individual):
        return self.parse_derivation_tree(decoded, individual)

    def parse_derivation_tree(self, derivation_tree, individual):
        """Parses the given derivation tree and returns the phenotype.

        :param derivation_tree: derivation tree of the individual
        :param individual: the individual itself
        :return: a correpsonding phenotype
        """
        raise NotImplementedError()


# noinspection PyAbstractClass
class GeTextFitness(GeFitness):
    """Base class for fitnesses that use a text as the decoding output.
    """
    def decode(self, individual):
        (text,
         finished,
         used_num,
         wraps,
         annotations) = self.grammar.to_text(individual.genotype, self.wraps)
        individual.set_annotations(annotations)
        individual.set_first_not_used(used_num)
        if not finished:
            raise GeFitness.NotFinishedError()

        return text