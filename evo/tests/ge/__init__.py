# -*- coding: utf8 -*-

import unittest

import evo.ge as ge
import evo.ge.support as ge_support

__author__ = 'Jan Žegklitz'


class TestGe(unittest.TestCase):
    def setUp(self):
        self.ge = ge.Ge(None, None, None, None, 'generational', 0)
        # <expr>  ::= (<expr><op><expr>) | <var> | <const>
        # <op>    ::= + | -
        # <var>   ::= x | y
        # <const> ::= 1

        # (x+(y-x))
        self.a = ge_support.CodonGenotypeIndividual([10, 11, 12, 13, 14, 15, 16,
                                                     17, 18, 19], 0)
        self.a.set_annotations([('expr', 10),  # 10
                                ('expr', 2),   # 11
                                ('var', 1),    # 12
                                ('op', 1),     # 13
                                ('expr', 6),   # 14
                                ('expr', 2),   # 15
                                ('var', 1),    # 16
                                ('op', 1),     # 17
                                ('expr', 2),   # 18
                                ('var', 1)])   # 19

        # ((y-x)+1)
        self.b = ge_support.CodonGenotypeIndividual([20, 21, 22, 23, 24, 25, 26,
                                                     27, 28, 29], 0)
        self.b.set_annotations([('expr', 10),   # 20
                                ('expr', 6),    # 21
                                ('expr', 2),    # 22
                                ('var', 1),     # 23
                                ('op', 1),      # 24
                                ('expr', 2),    # 25
                                ('var', 1),     # 26
                                ('op', 1),      # 27
                                ('expr', 2),    # 28
                                ('const', 1)])  # 29

    def test_subtree_crossover_hit(self):
        class Rng:
            def __init__(self):
                self.n = [1, 1]

            def randrange(self, _):
                return self.n.pop(0)

        self.ge.generator = Rng()
        ax, bx = self.ge.subtree_crossover(self.a, self.b)
        self.assertEqual([10, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17,
                          18, 19], ax.genotype)
        self.assertEqual([('expr', 10),
                          ('expr', 6),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 6),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 2),
                          ('var', 1)], ax.get_annotations())
        self.assertEqual([20, 11, 12, 27, 28, 29], bx.genotype)
        self.assertEqual([('expr', 10),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 2),
                          ('const', 1)], bx.get_annotations())

    def test_subtree_crossover_miss_hit(self):
        class Rng:
            def __init__(self):
                self.n = [9, 8, 0]

            def randrange(self, _):
                return self.n.pop(0)

        self.ge.generator = Rng()
        bx, ax = self.ge.subtree_crossover(self.b, self.a)
        self.assertEqual([20, 21, 22, 23, 24, 25, 26, 27, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19], bx.genotype)
        self.assertEqual([('expr', 10),
                          ('expr', 6),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 10),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 6),
                          ('expr', 2),
                          ('var', 1),
                          ('op', 1),
                          ('expr', 2),
                          ('var', 1)], bx.get_annotations())
        self.assertEqual([28, 29], ax.genotype)
        self.assertEqual([('expr', 2),
                          ('const', 1)], ax.get_annotations())

if __name__ == '__main__':
    unittest.main()
