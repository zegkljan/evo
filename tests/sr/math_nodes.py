# -*- coding: utf8 -*-

import unittest

import numpy as np

import evo.sr.math_nodes as mn


class TestBackpropagation(unittest.TestCase):
    def setUp(self):
        self.error_fn = lambda a, y: 1/2 * (a - y)**2
        self.error_derivative = lambda a, y: (a - y)

    def test_root(self):
        with self.subTest("exp"):
            root = mn.Exp()
            inp = mn.Const(None)

            root.children = [inp]
            inp.parent = root
            inp.parent_index = 0

            root.bias = None
            root.weights = np.array([0])

            values = [-np.e, -1, 0, 1, np.e]
            cases = [(x, a, b, y)
                     for x in values
                     for a in values
                     for b in values
                     for y in values]
            for x, a, b, y in cases:
                with self.subTest(x=x, a=a, b=b, y=y):
                    root.bias = b
                    root.weights = np.array([a])
                    inp.data = x
                    inp.notify_child_changed(0)
                    root.eval()
                    mn.backpropagate(root, self.error_derivative, y, {})
                    self.assertAlmostEqual(
                        np.exp(a * x + b) * (np.exp(a * x + b) - y),
                        root.d_bias,
                        6
                    )
                    self.assertAlmostEqual(
                        x * np.exp(a * x + b) * (np.exp(a * x + b) - y),
                        root.d_weights[0],
                        6
                    )

    def test_non_root(self):
        with self.subTest("correct output"):
            root = mn.Exp()
            mid = mn.Exp()
            inp = mn.Const(0)
            root.children = [mid]
            mid.parent = root
            mid.parent_index = 0
            mid.children = [inp]
            inp.parent = mid
            inp.parent_index = 0
            root.eval()

            mn.backpropagate(root, self.error_derivative, np.e)
            self.assertAlmostEqual(0, root.delta)
            self.assertAlmostEqual(0, mid.delta)

        with self.subTest("correct output"):
            root = mn.Exp()
            mid = mn.Exp()
            inp = mn.Const(0)
            root.children = [mid]
            mid.parent = root
            mid.parent_index = 0
            mid.children = [inp]
            inp.parent = mid
            inp.parent_index = 0
            root.eval()

            mn.backpropagate(root, self.error_derivative, 0)
            self.assertAlmostEqual(np.e, root.delta)
            self.assertAlmostEqual(np.e, mid.delta)


if __name__ == '__main__':
    unittest.main()
