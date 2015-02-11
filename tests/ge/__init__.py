# -*- coding: utf8 -*-

import unittest

import evo.ge as ge
import evo.ge.support as ge_support

__author__ = 'Jan Žegklitz'


class TestGe(unittest.TestCase):
    def test_subtree_crossover(self):
        # <expr> ::= (<expr><op><expr>) | <var>
        # <op>   ::= + | -
        # <var>  ::= x | y

        # (x+(y-x))
        i1 = ge_support.CodonGenotypeIndividual([10, 11, 12, 13, 14, 15, 16, 17,
                                                 18, 19], 0)
        i1.set_annotations([('expr', 10),
                            ('expr', 2),
                            ('var', 1),
                            ('op', 1),
                            ('expr', 6),
                            ('expr', 2),
                            ('var', 1),
                            ('op', 1),
                            ('expr', 2),
                            ('var', 1)])

        # ((y-x)+y)
        i1 = ge_support.CodonGenotypeIndividual([20, 21, 22, 23, 24, 25, 26, 27,
                                                 28, 29], 0)
        i1.set_annotations([('expr', 10),
                            ('expr', 6),
                            ('expr', 2),
                            ('var', 1),
                            ('op', 1),
                            ('expr', 2),
                            ('var', 1),
                            ('op', 1),
                            ('expr', 2),
                            ('var', 1)])

if __name__ == '__main__':
    unittest.main()
