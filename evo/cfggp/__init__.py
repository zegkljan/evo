# -*- coding: utf8 -*-
""" TODO docstring
"""

import multiprocessing.context
import random
import gc

import evo
import evo.ge

__author__ = 'Jan Žegklitz'


class DerivationTreeIndividual(evo.Individual):
    """A class representing an individual as a derivation tree.
    """

    def __init__(self, tree):
        """Creates the individual.

        :param tree: the derivation tree of the individual
        """
        evo.Individual.__init__(self)
        self.tree = tree
        self.fitness = None

    def __str__(self):
        return str(self.tree)

    def copy(self, carry_evaluation=True, carry_data=True):
        clone = DerivationTreeIndividual(self.tree.clone())
        evo.Individual.copy_evaluation(self, clone, carry_evaluation)
        evo.Individual.copy_data(self, clone, carry_data)
        return clone

    def get_tree(self):
        return self.tree


class Cfggp(multiprocessing.context.Process):
    """This class forms the CFG-GP algorithm.
    """

    class _GenerationsStop(object):

        def __init__(self, generations):
            self.generations = generations

        def __call__(self, cfggp):
            if cfggp.mode == 'generational':
                return cfggp.iterations >= self.generations
            if cfggp.mode == 'steady-state':
                return cfggp.iterations >= (self.generations *
                                            cfggp.pop_size / 2)

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
        :type fitness: :class:`evo.Fitness`
        :param int pop_size: size of the population; this value will be
            passed to the ``population_initializer``'s method ``initialize``()
        :param population_initializer: initializer used to initialize the
            initial population
        :type population_initializer:
            :class:`ge.init.PopulationInitializer`
        :param grammar: grammar this algorithm operates on
        :type grammar: :class:`evo.support.grammar.Grammar`
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
        :param stats: stats saving class
        :type stats: :class:`evo.support.Stats`
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
            self.stop = Cfggp._GenerationsStop(stop)
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
        """Runs the CFG-GP algorithm.
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

                self.crossover(o1, o2)

                #self.mutate(o1)
                #self.mutate(o2)

                o1 = self.derivation_to_codon(o1)
                o2 = self.derivation_to_codon(o2)

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

            self.crossover(o1, o2)

            #self.mutate(o1)
            #self.mutate(o2)

            o1 = self.derivation_to_codon(o1)
            o2 = self.derivation_to_codon(o2)

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

        #self.generator.shuffle(self.population)
        self.population.sort(key=lambda x: x.get_fitness(),
                             reverse=self.fitness.maximize())
        return self.population[0:self.elites_num]

    def select_tournament(self, population):
        candidates = self.generator.sample(population, self.tournament_size)
        best = None
        for candidate in candidates:
            if isinstance(candidate, evo.ge.CodonGenotypeIndividual):
                candidate = self.codon_to_derivation(candidate)
            if candidate.get_fitness() is None:
                self.evaluate(candidate)

            if best is None or self.fitness.compare(candidate.get_fitness(),
                                                    best.get_fitness()):
                best = candidate
        return best

    def crossover(self, p1, p2):
        if self.generator.random() < self.crossover_prob:
            self.subtree_crossover(p1, p2)

    def subtree_crossover(self, o1, o2):
        if not isinstance(o1, DerivationTreeIndividual):
            raise TypeError('Parent must be of type DerivationTreeIndividual.')
        if not isinstance(o2, DerivationTreeIndividual):
            raise TypeError('Parent must be of type DerivationTreeIndividual.')

        pred1 = lambda node: not node.is_leaf()
        n1 = o1.tree.get_filtered_subtree_size(pred1)
        point1 = self.generator.randrange(n1)
        node1 = o1.tree.get_filtered_nth_node(point1, pred1)

        pred2 = lambda node: not node.is_leaf() and node.data == node1.data
        n2 = o2.tree.get_filtered_subtree_size(pred2)
        while n2 == 0:
            n1 = o1.tree.get_filtered_subtree_size(pred1)
            point1 = self.generator.randrange(n1)
            node1 = o1.tree.get_filtered_nth_node(point1, pred1)

            pred2 = lambda node: (not node.is_leaf() and
                                  node.data == node1.data)
            n2 = o2.tree.get_filtered_subtree_size(pred2)

        point2 = self.generator.randrange(n2)
        node2 = o2.tree.get_filtered_nth_node(point2, pred2)

        if node1.parent is None and node2.parent is None:
            return
        elif node1.parent is None and node2.parent is not None:
            node1.parent = node2.parent
            node1.parent_index = node2.parent_index
            node1.parent.children[node1.parent_index] = node1
            node2.parent = None
            node2.parent_index = None
        elif node1.parent is not None and node2.parent is None:
            node2.parent = node1.parent
            node2.parent_index = node1.parent_index
            node2.parent.children[node2.parent_index] = node2
            node1.parent = None
            node1.parent_index = None
            o2.tree = node1
        else:
            node1.parent, node2.parent = node2.parent, node1.parent
            node1.parent_index, node2.parent_index = (node2.parent_index,
                                                      node1.parent_index)
            node1.parent.children[node1.parent_index] = node1
            node2.parent.children[node2.parent_index] = node2
        o1.set_fitness(None)
        o2.set_fitness(None)

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
        while l < u and c != l and c != u:
            ci = self.population[c]
            if self.fitness.compare(ci.get_fitness(), o.get_fitness()):
                l = c
            elif self.fitness.compare(o.get_fitness(), ci.get_fitness()):
                u = c
            else:
                break
            c = (l + u) // 2
        self.population.insert(c + 1, o)

    def codon_to_derivation(self, codon):
        if not isinstance(codon, evo.ge.CodonGenotypeIndividual):
            raise TypeError

        (tree, _, _, _) = self.grammar.to_tree(decisions=codon.genotype,
                                               max_wraps=0,
                                               max_depth=float('inf'))
        derivation = DerivationTreeIndividual(tree)
        derivation.set_fitness(codon.get_fitness())
        return derivation

    def derivation_to_codon(self, derivation):
        if not isinstance(derivation, DerivationTreeIndividual):
            raise TypeError('Individual must be of type '
                            'evo.cfggp.DerivationTreeIndividual.')
        choices, _ = \
            self.grammar.derivation_tree_to_choice_sequence(derivation.tree)

        codon = evo.ge.CodonGenotypeIndividual(choices, None)
        codon.set_fitness(derivation.get_fitness())
        return codon
