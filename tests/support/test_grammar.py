# -*- coding: utf8 -*-
"""
Created on Apr 8, 2014

.. moduleauthor:: Jan Žegklitz <zegkljan@gmail.com>
"""
import unittest
import evo.support.grammar as grammar
import evo.support.tree as tree


class GrammarTest(unittest.TestCase):

    def test_grammar(self):
        grammar_dict = {"start-rule": "start",
                        "rules": {"start": [[("N", "expr")]],
                                  "expr": [[("N", "par-expr"),
                                            ("T", "+"),
                                            ("N", "par-expr")],
                                           [("N", "num"),
                                            ("T", "+"),
                                            ("N", "num")],
                                           [("N", "num")]],
                                  "par-expr": [[("T", "("),
                                                ("N", "expr"),
                                                ("T", ")")]],
                                  "num": [[("T", "a")],
                                          [("T", "b")]],
                                  "orphan": [[("T", "orphan-term")]]
                                   }
                        }
        g = grammar.Grammar(grammar_dict)
        sr = g.get_start_rule()

        # start
        start = sr
        self.assertTrue(type(start) is grammar.Rule)
        self.assertEqual("start", start.name)
        self.assertEqual(1, len(start.get_choices()))
        self.assertEqual(1, len(start.get_choice(0)))

        # start - expr
        expr = start.get_choice(0)[0]
        self.assertTrue(type(expr) is grammar.Rule)
        self.assertEqual("expr", expr.name)
        self.assertEqual(3, len(expr.get_choices()))
        self.assertEqual(3, len(expr.get_choice(0)))
        self.assertEqual(3, len(expr.get_choice(1)))
        self.assertEqual(1, len(expr.get_choice(2)))

        # start - expr[0] - par-expr
        par_expr = expr.get_choice(0)[0]
        self.assertTrue(type(par_expr) is grammar.Rule)
        self.assertEqual("par-expr", par_expr.name)
        self.assertEqual(1, len(par_expr.get_choices()))
        self.assertEqual(3, len(par_expr.get_choice(0)))

        # start - expr[0] - par-expr[0] - (
        lpar = par_expr.get_choice(0)[0]
        self.assertTrue(type(lpar) is grammar.Terminal)
        self.assertEqual("(", lpar.text)

        # start - expr[0] - par-expr[0] - expr
        expr2 = par_expr.get_choice(0)[1]
        self.assertTrue(expr is expr2)

        # start - expr[0] - par-expr[0] - )
        rpar = par_expr.get_choice(0)[2]
        self.assertTrue(type(rpar) is grammar.Terminal)
        self.assertEqual(")", rpar.text)

        # start - expr[0] - +
        plus = expr.get_choice(0)[1]
        self.assertTrue(type(plus) is grammar.Terminal)
        self.assertEqual("+", plus.text)

        # start - expr[0] - par-expr
        par_expr2 = expr.get_choice(0)[0]
        self.assertTrue(par_expr is par_expr2)

        # start - expr[1] - num
        num = expr.get_choice(1)[0]
        self.assertTrue(type(num) is grammar.Rule)
        self.assertEqual("num", num.name)
        self.assertEqual(2, len(num.get_choices()))
        self.assertEqual(1, len(num.get_choice(0)))
        self.assertEqual(1, len(num.get_choice(1)))

        # start - expr[1] - num[0] - a
        a = num.get_choice(0)[0]
        self.assertTrue(type(a) is grammar.Terminal)
        self.assertEqual("a", a.text)

        # start - expr[1] - num[1] - b
        b = num.get_choice(1)[0]
        self.assertTrue(type(b) is grammar.Terminal)
        self.assertEqual("b", b.text)

        # start - expr[1] - +
        plus2 = expr.get_choice(1)[1]
        self.assertTrue(plus is plus2)

        # start - expr[1] - num
        num2 = expr.get_choice(1)[2]
        self.assertTrue(num is num2)

        # start - expr[2] - num
        num3 = expr.get_choice(2)[0]
        self.assertTrue(num is num3)

    def test_terminality(self):
        gr = {'start-rule': '<Code>',
              'rules': {'<Code>': [[('N', '<Line>')],
                                   [('T', 'prog2'), ('N', '<Line>'),
                                    ('N', '<Code>')]],
                        '<Line>': [[('N', '<Condition>')],
                                   [('N', '<Action>')]],
                        '<Action>': [[('T', 'move')],
                                     [('T', 'right')],
                                     [('T', 'left')]],
                        '<Condition>': [[('T', 'iffoodahead'), ('N', '<Code>'),
                                         ('N', '<Code>')]]}}
        g = grammar.Grammar(gr)

        code = g.get_start_rule()
        line = code.get_choice(0)[0]
        condition = line.get_choice(0)[0]
        action = line.get_choice(1)[0]

        self.assertIsNot(code, line)
        self.assertIsNot(code, condition)
        self.assertIsNot(code, action)
        self.assertIsNot(line, condition)
        self.assertIsNot(line, action)
        self.assertIsNot(condition, action)

        self.assertEqual([line], code.get_terminal_choice(0))
        self.assertEqual(1, code.get_terminal_choices_num())

        self.assertEqual([code, code], condition.get_terminal_choice(0)[1:])
        self.assertEqual(1, condition.get_terminal_choices_num())

        self.assertEqual([action], line.get_terminal_choice(0))
        self.assertEqual(1, line.get_terminal_choices_num())

        self.assertEqual(3, action.get_terminal_choices_num())

    def test_to_dict(self):
        grammar_dict = {"start-rule": "start",
                        "rules": {"start": [[("N", "expr")]],
                                  "expr": [[("N", "par-expr"),
                                            ("T", "+"),
                                            ("N", "par-expr")],
                                           [("N", "num"),
                                            ("T", "+"),
                                            ("N", "num")],
                                           [("N", "num")]],
                                  "par-expr": [[("T", "("),
                                                ("N", "expr"),
                                                ("T", ")")]],
                                  "num": [[("T", "a")],
                                          [("T", "b")]]
                                   }
                        }
        g1 = grammar.Grammar(grammar_dict)
        self.assertEqual(grammar_dict, g1.to_dict())

        g2 = grammar.Grammar(g1.to_dict())
        self.assertEqual(g1.to_dict(), g2.to_dict())

        grammar_dict = {"start-rule": "start",
                        "rules": {"start": [[("N", "expr")]],
                                  "expr": [[("N", "par-expr"),
                                            ("T", "+"),
                                            ("N", "par-expr")],
                                           [("N", "num"),
                                            ("T", "+"),
                                            ("N", "num")],
                                           [("N", "num")]],
                                  "par-expr": [[("T", "("),
                                                ("N", "expr"),
                                                ("T", ")")]],
                                  "num": [[("T", "a")],
                                          [("T", "b")]],
                                  "orphan": [[("T", "orphan-term")]]
                                   }
                        }
        g1 = grammar.Grammar(grammar_dict)
        self.assertNotEqual(grammar_dict, g1.to_dict())

    def test_to_tree(self):
        grammar_dict = {"start-rule": "start",
                        "rules": {"start": [[("N", "expr")]],
                                  "expr": [[("N", "num"),
                                            ("N", "op"),
                                            ("N", "num")],
                                           [("N", "expr"),
                                            ("N", "op"),
                                            ("N", "num")],
                                           [("N", "num"),
                                            ("N", "op"),
                                            ("N", "expr")],
                                           [("N", "expr"),
                                            ("N", "op"),
                                            ("N", "expr")],
                                           [("N", "num")]],
                                  "num": [[("T", "0"),
                                           ("N", "num")],
                                          [("T", "1"),
                                           ("N", "num")],
                                          [("T", "2"),
                                           ("N", "num")],
                                          [("T", "0")],
                                          [("T", "1")],
                                          [("T", "2")]],
                                  "op": [[("T", "+")],
                                         [("T", "*")]]
                                   }
                        }
        g = grammar.Grammar(grammar_dict)

        # 10+102
        startNode = tree.TreeNode(None, None, [], "start")
        exprNode = tree.TreeNode(startNode, 0, [], "expr")
        numNode1 = tree.TreeNode(exprNode, 0, [], "num")
        termNode1 = tree.TreeNode(numNode1, 0, [], "1")
        numNode2 = tree.TreeNode(numNode1, 1, [], "num")
        termNode2 = tree.TreeNode(numNode2, 0, [], "0")
        opNode = tree.TreeNode(exprNode, 1, [], "op")
        termNode3 = tree.TreeNode(opNode, 0, [], "+")
        numNode3 = tree.TreeNode(exprNode, 2, [], "num")
        termNode4 = tree.TreeNode(numNode3, 0, [], "1")
        numNode4 = tree.TreeNode(numNode3, 1, [], "num")
        termNode5 = tree.TreeNode(numNode4, 0, [], "0")
        numNode5 = tree.TreeNode(numNode4, 1, [], "num")
        termNode6 = tree.TreeNode(numNode5, 0, [], "2")

        startNode.children = [exprNode]
        exprNode.children = [numNode1, opNode, numNode3]
        numNode1.children = [termNode1, numNode2]
        termNode1.children = None
        numNode2.children = [termNode2]
        termNode2.children = None
        opNode.children = [termNode3]
        termNode3.children = None
        numNode3.children = [termNode4, numNode4]
        termNode4.children = None
        numNode4.children = [termNode5, numNode5]
        termNode5.children = None
        numNode5.children = [termNode6]
        termNode6.children = None

        out = g.to_tree([0, 1, 3, 0, 1, 0, 5, 11], max_wraps=0)
        self.assertEqual(startNode.__str__(), out[0].__str__())
        self.assertEqual(True, out[1])
        self.assertEqual(7, out[2])
        self.assertEqual(0, out[3])
        self.assertEqual([('expr', 7),
                          ('num', 2),
                          ('num', 1),
                          ('op', 1),
                          ('num', 3),
                          ('num', 2),
                          ('num', 1)], out[4])

    def test_to_tree2(self):
        grammar_dict = {"start-rule": "start",
                        "rules": {"start": [[("N", "expr")]],
                                  "expr": [[("N", "num"),
                                            ("N", "op"),
                                            ("N", "num")],
                                           [("N", "expr"),
                                            ("N", "op"),
                                            ("N", "num")],
                                           [("N", "num"),
                                            ("N", "op"),
                                            ("N", "expr")],
                                           [("N", "expr"),
                                            ("N", "op"),
                                            ("N", "expr")],
                                           [("N", "num")]],
                                  "num": [[("T", "0"),
                                           ("N", "num")],
                                          [("T", "1"),
                                           ("N", "num")],
                                          [("T", "2"),
                                           ("N", "num")],
                                          [("T", "0")],
                                          [("T", "1")],
                                          [("T", "2")]],
                                  "op": [[("T", "+")],
                                         [("T", "*")]]
                                   }
                        }
        g = grammar.Grammar(grammar_dict)

        # 10<op><num>
        startNode = tree.TreeNode(None, None, [], "start")
        exprNode = tree.TreeNode(startNode, 0, [], "expr")
        numNode1 = tree.TreeNode(exprNode, 0, [], "num")
        termNode1 = tree.TreeNode(numNode1, 0, [], "1")
        numNode2 = tree.TreeNode(numNode1, 1, [], "num")
        termNode2 = tree.TreeNode(numNode2, 0, [], "0")
        opNode = tree.TreeNode(exprNode, 1, [], "op")
        numNode3 = tree.TreeNode(exprNode, 2, [], "num")

        startNode.children = [exprNode]
        exprNode.children = [numNode1, opNode, numNode3]
        numNode1.children = [termNode1, numNode2]
        termNode1.children = None
        numNode2.children = [termNode2]
        termNode2.children = None
        opNode.children = []
        numNode3.children = []

        out = g.to_tree([0, 1, 3], max_wraps=0)
        self.assertEqual(startNode.__str__(), out[0].__str__())
        self.assertEqual(False, out[1])
        self.assertEqual(3, out[2])
        self.assertEqual(0, out[3])
        self.assertIsNone(out[4])

    def test_to_tree3(self):
        grammar_dict = {'start-rule': '<E>',
                        'rules': {'<E>': [[('T', 'move')],
                                          [('T', 'left')],
                                          [('T', 'right')],
                                          [('T', 'iffoodahead'), ('N', '<E>'),
                                           ('N', '<E>')],
                                          [('T', 'prog2'), ('N', '<E>'),
                                           ('N', '<E>')]]}}
        g = grammar.Grammar(grammar_dict)

        # [3, 0] -> (<E> iffoodahead (<E> move) (<E>))
        eNode1 = tree.TreeNode(None, None, [], "<E>")
        iffoodaheadNode = tree.TreeNode(eNode1, 0, None, "iffoodahead")
        eNode2 = tree.TreeNode(eNode1, 1, [], "<E>")
        moveNode = tree.TreeNode(eNode1, 0, None, "move")
        eNode3 = tree.TreeNode(eNode1, 2, [], "<E>")

        eNode1.children = [iffoodaheadNode, eNode2, eNode3]
        iffoodaheadNode.children = None
        eNode2.children = [moveNode]
        moveNode.children = None
        eNode3.children = []

        out = g.to_tree([3, 0], max_wraps=0)
        self.assertEqual(eNode1.__str__(), out[0].__str__())
        self.assertEqual(False, out[1])
        self.assertEqual(2, out[2])
        self.assertEqual(0, out[3])
        self.assertIsNone(out[4])

    def test_to_text(self):
        grammar_dict = {"start-rule": "start",
                        "rules": {"start": [[("N", "expr")]],
                                  "expr": [[("N", "num"),
                                            ("N", "op"),
                                            ("N", "num")],
                                           [("N", "expr"),
                                            ("N", "op"),
                                            ("N", "num")],
                                           [("N", "num"),
                                            ("N", "op"),
                                            ("N", "expr")],
                                           [("N", "expr"),
                                            ("N", "op"),
                                            ("N", "expr")],
                                           [("N", "num")]],
                                  "num": [[("T", "0"),
                                           ("N", "num")],
                                          [("T", "1"),
                                           ("N", "num")],
                                          [("T", "2"),
                                           ("N", "num")],
                                          [("T", "0")],
                                          [("T", "1")],
                                          [("T", "2")]],
                                  "op": [[("T", "+")],
                                         [("T", "*")]]
                                   }
                        }
        g = grammar.Grammar(grammar_dict)

        out = g.to_text([0, 1, 3, 0, 1, 0, 5, 11], max_wraps=0)
        self.assertEqual('10+102', out[0])
        self.assertEqual(True, out[1])
        self.assertEqual(7, out[2])
        self.assertEqual(0, out[3])
        self.assertEqual([('expr', 7),
                          ('num', 2),
                          ('num', 1),
                          ('op', 1),
                          ('num', 3),
                          ('num', 2),
                          ('num', 1)], out[4])

    def test_derivation_tree_to_text(self):
        # 10+102
        startNode = tree.TreeNode(None, None, [], "start")
        exprNode = tree.TreeNode(startNode, 0, [], "expr")
        numNode1 = tree.TreeNode(exprNode, 0, [], "num")
        termNode1 = tree.TreeNode(numNode1, 0, [], "1")
        numNode2 = tree.TreeNode(numNode1, 1, [], "num")
        termNode2 = tree.TreeNode(numNode2, 0, [], "0")
        opNode = tree.TreeNode(exprNode, 1, [], "op")
        termNode3 = tree.TreeNode(opNode, 0, [], "+")
        numNode3 = tree.TreeNode(exprNode, 2, [], "num")
        termNode4 = tree.TreeNode(numNode3, 0, [], "1")
        numNode4 = tree.TreeNode(numNode3, 1, [], "num")
        termNode5 = tree.TreeNode(numNode4, 0, [], "0")
        numNode5 = tree.TreeNode(numNode4, 1, [], "num")
        termNode6 = tree.TreeNode(numNode5, 0, [], "2")

        startNode.children = [exprNode]
        exprNode.children = [numNode1, opNode, numNode3]
        numNode1.children = [termNode1, numNode2]
        termNode1.children = None
        numNode2.children = [termNode2]
        termNode2.children = None
        opNode.children = [termNode3]
        termNode3.children = None
        numNode3.children = [termNode4, numNode4]
        termNode4.children = None
        numNode4.children = [termNode5, numNode5]
        termNode5.children = None
        numNode5.children = [termNode6]
        termNode6.children = None

        self.assertEqual("10+102", grammar.derivation_tree_to_text(startNode))

if __name__ == "__main__":
    unittest.main()
