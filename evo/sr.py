# -*- coding: utf8 -*-
"""This module implements various symbolic regression solvers.
"""

import numpy
import numpy.linalg
import numpy.matlib
import copy
import functools

import evo
import evo.ge
import evo.utils.tree
import evo.utils.grammar

__author__ = 'Jan Žegklitz'


class MultiGeneGeSrFitness(evo.ge.GeTreeFitness):
    """A fitness for symbolic regression that operates with multiple genes at
    the grammar level.

    The evaluation of the genotypes uses multi-gene symbolic regression approach
    [MGGP].

    The multi-geneneness is achieved by encapsulating the given grammar into
    another grammar that effectively produces multiple genes. The maximum number
    of the genes is specified by constructor argument. For the details about
    this procedure see :meth:`.encapsulate_grammar`.

    .. [MGGP] TODO MGGP reference
    """

    MULTIGENE_START = 'multigene-start'
    GENE = 'gene'

    def __init__(self, grammar, max_genes, unfinished_fitness, error_fitness,
                 handled_errors, target, wraps=0, skip_if_evaluated=True):
        """
        :param grammar: the base grammar
        :param max_genes: maximum number of genes
        :param unfinished_fitness: fitness to assign to individuals that did not
            finished transcription
        :param target: target values of the datapoints: an N x 1 matrix where N
            is the number of datapoints
        :param wraps: number of times the wrapping (reusing the codon sequence
            from beginning) is allowed to happen
        :param skip_if_evaluated: if ``True`` then already evaluated individuals
            will not be evaluated again and their stored fitness will be used
        """
        evo.ge.GeTreeFitness.__init__(
            self, MultiGeneGeSrFitness.encapsulate_grammar(grammar, max_genes),
            unfinished_fitness, wraps, skip_if_evaluated)
        self.error_fitness = error_fitness
        self.target = numpy.matrix(target, copy=False)
        self.errors = tuple([ZeroDivisionError, FloatingPointError] +
                            handled_errors)

    def evaluate_phenotype(self, phenotype, individual):
        metafeatures = None
        gene_trees = []
        i = 1
        for gene_tree, subphenotype in phenotype:
            gene_trees.append(gene_tree)
            try:
                result = self.apply_gene_phenotype(subphenotype)
            except self.errors as e:
                return self.error_fitness
            if metafeatures is None:
                # noinspection PyTypeChecker
                metafeatures = numpy.matlib.ones((result.shape[0],
                                                  len(phenotype) + 1))
            metafeatures[:, i] = numpy.asmatrix(result).T
            i += 1

        if numpy.any(numpy.logical_or(numpy.isinf(metafeatures),
                                      numpy.isnan(metafeatures))):
            return self.error_fitness
        try:
            weights = (numpy.linalg.pinv(metafeatures.T * metafeatures) *
                       metafeatures.T * self.target)
        except numpy.linalg.linalg.LinAlgError:
            return self.error_fitness
        target_estimate = metafeatures * weights
        error = self.target - target_estimate
        # just in case we need to check whether the target estimate corresponds
        # to the one computed by using the full compound predictor
        # weights2 = numpy.squeeze(numpy.asarray(weights))
        # compound_phenotype = self.combine_gene_derivation_trees(gene_trees,
        #                                                         weights2[1:],
        #                                                         weights2[0])
        # target_estimate2 = numpy.apply_along_axis(compound_phenotype, 1,
        #                                           self.features)
        individual.set_data('weights', list(numpy.asarray(weights).squeeze()))
        return (error.T * error)[0, 0]

    def apply_gene_phenotype(self, phenotype):
        """Applies the given phenotype to get its "metafeature".

        This method is responsible for applying the given phenotype of one gene
        to some data and return the output.

        :param phenotype: the phenotype to apply
        :return: a column vector of computed values
        """
        raise NotImplementedError()

    def parse_derivation_tree(self, derivation_tree, individual):
        assert derivation_tree.data == MultiGeneGeSrFitness.MULTIGENE_START
        genes = [None] * len(derivation_tree.children)
        for i in range(len(derivation_tree.children)):
            gene_tree = derivation_tree.children[i]
            genes[i] = (gene_tree, self.parse_gene_derivation_tree(gene_tree,
                                                                   individual))
        return genes

    def parse_gene_derivation_tree(self, derivation_tree, individual):
        """Should return a callable object that takes the number of variables
        equal to the number of features and returns a scalar.

        :param derivation_tree: a derivation tree of a single gene
        :param individual: a reference to the original individual
        """
        raise NotImplementedError()

    def combine_gene_derivation_trees(self, derivation_trees, coefficients,
                                      bias):
        """Should return a callable object that takes the number of variables
        equal to the number of features and returns a scalar.

        The arguments ``derivation_trees`` and ``coefficients`` contain the
        derivation trees of the genes and their optimal coefficients
        respectively. The ``bias`` argument is technically a coefficient of the
        unit component.

        :param derivation_trees: a list of derivation trees of the genes
        :param coefficients: a list of coefficients of the particular genes
        :param bias: the additive component of the compound model
        """
        raise NotImplementedError()

    def sort(self, population, reverse=False, *args):
        population.sort(key=functools.cmp_to_key(self.compare))
        return True

    def compare(self, i1, i2, *args):
        f1 = i1.get_fitness()
        f2 = i2.get_fitness()
        if f1 is None:
            self.evaluate(i1)
            f1 = i1.get_fitness()
        if f2 is None:
            self.evaluate(i2)
            f2 = i1.get_fitness()
        return f1 < f2

    @staticmethod
    def encapsulate_grammar(grammar, max_genes, force_consume=True):
        """Encapsulates the given grammar into a multi-gene grammar.

        * The grammar is extended by a nonterminal named ``gene`` and with a
          rule ``<gene> ::= <start>`` where ``start`` is the original start
          nonterminal of the base grammar.
        * The grammar is extended by a nonterminal named ``multigene-start``
          and a rule that transcribes this new nonterminal to a sequence of
          repeated ``gene`` nonterminals of length ranging from 1 to
          ``max_genes``.
        * The start nonterminal of the base grammar is set to a different
          nonterminal named ``multigene-start``.

        If ``force_consume`` is set to ``True`` then the ``gene`` nonterminal
        will force the codon transcription process to consume a codon even it
        would normally wouldn't because there is only one choice. This is
        achieved by making the rule having 2 choices which are identical.

        It is important that no nonterminals in the base grammar are named
        ``multigene-start`` and ``gene``.

        .. admonition:: Example

            Suppose the following grammar is given::

                <expr> ::= <expr>-<expr>
                         | <expr>/<expr>
                         | x
                         | 1

            Then the encapsulated grammar will look like::

                <multigene-start> ::= <gene>
                                    | <gene><gene>
                                    | <gene><gene><gene>
                                    ...
                                    | <gene><gene>...<gene><gene>
                           <gene> ::= <expr>[ | <expr>]
                           <expr> ::= <expr>-<expr>
                                    | <expr>/<expr>
                                    | x
                                    | 1

            The part in square brackets in the ``<gene>`` rule is present only
            if ``force_consume`` is ``True``.

        :param grammar: base grammar
        :param max_genes: maximum number of genes
        :param bool force_consume: whether the ``<gene>`` rule should force
            consuming a codon (default is ``True``)
        :return: a dictionary representing the grammar that can be directly fed
            to :class:`evo.utils.grammar.Grammar` constructor.
        """
        if isinstance(grammar, evo.utils.grammar.Grammar):
            grammar_dict = grammar.to_dict()
        else:
            grammar_dict = copy.deepcopy(grammar)

        # add gene -> start-rule
        if force_consume:
            grammar_dict['rules'][MultiGeneGeSrFitness.GENE] = \
                [[('N', grammar_dict['start-rule'])]] * 2
        else:
            grammar_dict['rules'][MultiGeneGeSrFitness.GENE] = \
                [[('N', grammar_dict['start-rule'])]]

        # add multigene-start -> gene
        multigene_choices = [None] * max_genes
        for i in range(max_genes):
            multigene_choices[i] = [('N', MultiGeneGeSrFitness.GENE)] * (i + 1)
        grammar_dict['rules'][MultiGeneGeSrFitness.MULTIGENE_START] = \
            multigene_choices

        # set new start-rule to multigene-start
        grammar_dict['start-rule'] = MultiGeneGeSrFitness.MULTIGENE_START

        return grammar_dict


class MultiGeneGe(evo.ge.Ge):
    """This class represents the GE algorithm modified to work in the multi-gene
    genetic programming (MGGP) fashion.
    """
    def __init__(self, fitness, pop_size, population_initializer, grammar, mode,
                 stop, name=None, **kwargs):
        """The constructor is identical to the one of :class:`evo.ge.Ge` except
        for the ``crossover_type`` keyword argument which is extended as
        described below.

        The ``crossover_type`` keyword argument can have these additional
        values:

            * ``('cr-high-level', gene_rule, crossover_rate, g_max)`` - the
              rate-based high-level crossover (exchanges whole genes between
              individuals); ``gene_rule`` is expected to be the name of the rule
              which encapsulates the genes, ``crossover_rate`` is expected to be
              the probability of a gene being selected for crossover and
              ``g_max`` is the maximum number of genes
            * ``('low-level', hl_rules)`` - the low-level crossover (exchanges
              parts of two genes); the ``hl_rules`` is expected to be an
              iterable of names of rules which are on the "high" level
            * ``('probabilistic', (prob, method1), (prob, method2), ...)`` -
              a so-called probabilistic crossover - it is composed of multiple
              crossover methods (``method1``, ``method2`` etc.), exactly as
              described by other possible values of the ``crossover_type``
              argument (including those in the superclass' constructor), where
              each of the particular crossover methods as a probability
              (``prob``) of being performed. If the probabilities do not sum up
              to 1 they will be scaled so that they do.

        .. seealso: :class:`evo.ge.Ge`
        """
        super().__init__(fitness, pop_size, population_initializer, grammar,
                         mode, stop, name, **kwargs)

    def setup_crossover(self, crossover_type):
        if crossover_type[0] == 'cr-high-level':
            return self.cr_high_level_crossover, (crossover_type[1],
                                                  crossover_type[2],
                                                  crossover_type[3])
        elif crossover_type[0] == 'low-level':
            return self.low_level_crossover, (crossover_type[1], )
        elif crossover_type[0] == 'probabilistic':
            crossover_method = self.probabilistic_crossover
            probs_methods = []
            for prob, subcrossover in crossover_type[1:]:
                cm, cma = self.setup_crossover(subcrossover)
                if not probs_methods:
                    probs_methods.append([prob, (cm, cma)])
                else:
                    probs_methods.append([probs_methods[-1][0] + prob,
                                          (cm, cma)])
            for i in range(len(probs_methods)):
                probs_methods[i][0] = probs_methods[i][0] / probs_methods[-1][0]

            crossover_method_args = (tuple(probs_methods), )
            return crossover_method, crossover_method_args
        else:
            return super().setup_crossover(crossover_type)

    def cr_high_level_crossover(self, o1, o2, gene_rule, crossover_rate, g_max):
        """Performs the rate based high level crossover between the two parents.
        """
        if not isinstance(o1, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        if not isinstance(o2, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')

        if not o1.get_annotations() or not o2.get_annotations():
            return []

        g1 = o1.genotype
        g2 = o2.genotype

        if len(g1) == len(g2) == 1:
            return [o1, o2]

        a1 = list(enumerate(o1.get_annotations()))
        a2 = list(enumerate(o2.get_annotations()))
        fa1 = list(filter(lambda x: self._cr_hl_xover_filter(gene_rule,
                                                             crossover_rate, x),
                          a1))
        fa2 = list(filter(lambda x: self._cr_hl_xover_filter(gene_rule,
                                                             crossover_rate, x),
                          a2))

        if not fa1 and not fa2:
            return [o1, o2]

        assert g1, g1
        assert g2, g2

        g1_num = g1[0] % g_max + 1
        g2_num = g2[0] % g_max + 1
        g1_num_diff = len(fa2) - len(fa1)
        g2_num_diff = len(fa1) - len(fa2)

        g1_genes = []
        g1_annots = []
        for i, a in reversed(fa1):
            g1_genes = g1[i:i + a[1]] + g1_genes
            g1_annots = a1[i:i + a[1]] + g1_annots
            del g1[i:i + a[1]]
            del a1[i:i + a[1]]

        g2_genes = []
        g2_annots = []
        for i, a in reversed(fa2):
            g2_genes = g2[i:i + a[1]] + g2_genes
            g2_annots = a2[i:i + a[1]] + g2_annots
            del g2[i:i + a[1]]
            del a2[i:i + a[1]]

        g1 += g2_genes
        g2 += g1_genes

        a1 += g2_annots
        a2 += g1_annots

        g1_delete = max(0, g1_num + g1_num_diff - g_max)
        g2_delete = max(0, g2_num + g2_num_diff - g_max)

        ga1 = list(filter(lambda x: x[1] is not None and x[1][0] == gene_rule,
                          a1))
        ga2 = list(filter(lambda x: x[1] is not None and x[1][0] == gene_rule,
                          a2))

        for i, a in self.generator.sample(ga1, g1_delete):
            del g1[i:i + a[1]]
            del a1[i:i + a[1]]

        for i, a in self.generator.sample(ga2, g2_delete):
            del g2[i:i + a[1]]
            del a2[i:i + a[1]]

        g1[0] += g1_num_diff - g1_delete + g_max
        if g1[0] > o1.get_max_codon_value():
            g1[0] -= g_max

        g2[0] += g2_num_diff - g2_delete + g_max
        if g2[0] > o2.get_max_codon_value():
            g2[0] -= g_max

        o1.set_annotations(None)
        o2.set_annotations(None)

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]

    def _cr_hl_xover_filter(self, gene_rule, crossover_rate, i_annot):
        return (i_annot[1] is not None and i_annot[1][0] == gene_rule and
                self.generator.random() < crossover_rate)

    def low_level_crossover(self, o1, o2, hl_rules):
        if not isinstance(o1, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')
        if not isinstance(o2, evo.ge.support.CodonGenotypeIndividual):
            raise TypeError('Parent must be of type CodonGenotypeIndividual.')

        a1 = list(map(lambda x: (None if x is not None and x[0] in hl_rules
                                 else x),
                      o1.get_annotations()))
        a2 = list(map(lambda x: (None if x is not None and x[0] in hl_rules
                                 else x),
                      o2.get_annotations()))
        o1.set_annotations(None)
        o2.set_annotations(None)

        return super().subtree_crossover(o1, o2)

    def probabilistic_crossover(self, o1, o2, probs_methods):
        r = self.generator.random()
        method = None
        for p, m in probs_methods:
            method = m
            if p >= r:
                break
        return method[0](o1, o2, *method[1])