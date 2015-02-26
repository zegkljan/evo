# -*- coding: utf8 -*-

import unittest

import evo.sr as sr

__author__ = 'Jan Žegklitz'


class TestMultiGeneGeSrFitness(unittest.TestCase):
    def test_encapsulate_grammar(self):
        # <expr> ::= <expr>-<expr>
        #          | <expr>/<expr>
        #          | x
        #          | 1
        test_grammar_dict = {
            'start-rule': 'expr',
            'rules': {
                'expr': [[('N', 'expr'), ('T', '-'), ('N', 'expr')],
                         [('N', 'expr'), ('T', '/'), ('N', 'expr')],
                         [('T', 'x')],
                         [('T', '1')]]
            }
        }
        with self.subTest('force_consume=False'):
            # <multigene-start> ::= <gene> | <gene><gene> | <gene><gene><gene>
            #            <gene> ::= <expr>
            #            <expr> ::= <expr>-<expr>
            #                     | <expr>/<expr>
            #                     | x
            #                     | 1
            expected_grammar_dict = {
                'start-rule': 'multigene-start',
                'rules': {
                    'expr': [[('N', 'expr'), ('T', '-'), ('N', 'expr')],
                             [('N', 'expr'), ('T', '/'), ('N', 'expr')],
                             [('T', 'x')],
                             [('T', '1')]],
                    'gene': [[('N', 'expr')]],
                    'multigene-start': [[('N', 'gene')],
                                        [('N', 'gene')] * 2,
                                        [('N', 'gene')] * 3]
                }
            }
            self.assertEqual(expected_grammar_dict,
                             sr.MultiGeneGeSrFitness.encapsulate_grammar(
                                 test_grammar_dict, 3, force_consume=False))
        with self.subTest('force_consume=True'):
            # <multigene-start> ::= <gene> | <gene><gene> | <gene><gene><gene>
            #            <gene> ::= <expr> | <expr>
            #            <expr> ::= <expr>-<expr>
            #                     | <expr>/<expr>
            #                     | x
            #                     | 1
            expected_grammar_dict = {
                'start-rule': 'multigene-start',
                'rules': {
                    'expr': [[('N', 'expr'), ('T', '-'), ('N', 'expr')],
                             [('N', 'expr'), ('T', '/'), ('N', 'expr')],
                             [('T', 'x')],
                             [('T', '1')]],
                    'gene': [[('N', 'expr')],
                             [('N', 'expr')]],
                    'multigene-start': [[('N', 'gene')],
                                        [('N', 'gene')] * 2,
                                        [('N', 'gene')] * 3]
                }
            }
            self.maxDiff = None
            self.assertEqual(expected_grammar_dict,
                             sr.MultiGeneGeSrFitness.encapsulate_grammar(
                                 test_grammar_dict, 3, force_consume=True))


if __name__ == '__main__':
    unittest.main()
