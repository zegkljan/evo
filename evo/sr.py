# -*- coding: utf8 -*-
"""This module implements various symbolic regression solvers.
"""

import numpy
import numpy.linalg
import numpy.matlib
import copy

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

    def __init__(self, grammar, max_genes, unfinished_fitness, target, wraps=0,
                 skip_if_evaluated=True):
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
        self.target = numpy.matrix(target, copy=False)

    def evaluate_phenotype(self, phenotype, individual):
        metafeatures = None
        gene_trees = []
        i = 1
        for gene_tree, subphenotype in phenotype:
            gene_trees.append(gene_tree)
            result = self.apply_gene_phenotype(subphenotype)
            if metafeatures is None:
                # noinspection PyTypeChecker
                metafeatures = numpy.matlib.ones((result.shape[0],
                                                  len(phenotype) + 1))
            metafeatures[:, i] = numpy.asmatrix(result).T
            i += 1

        weights = (numpy.linalg.pinv(metafeatures.T * metafeatures) *
                   metafeatures.T * self.target)
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

    def parse_derivation_tree(self, derivation_tree):
        assert derivation_tree.data == MultiGeneGeSrFitness.MULTIGENE_START
        genes = [None] * len(derivation_tree.children)
        for i in range(len(derivation_tree.children)):
            gene_tree = derivation_tree.children[i]
            genes[i] = (gene_tree, self.parse_gene_derivation_tree(gene_tree))
        return genes

    def parse_gene_derivation_tree(self, derivation_tree):
        """Should return a callable object that takes the number of variables
        equal to the number of features and returns a scalar.

        :param derivation_tree: a derivation tree of a single gene
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
        return False
        # def compare(i1, i2):
        #     return
        #
        # compare = lambda i1, i2: i2.get_fitness()!!!!

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