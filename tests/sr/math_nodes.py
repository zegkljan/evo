# -*- coding: utf8 -*-

import unittest

import numpy as np

import evo.sr.math_nodes as mn


class Identity(mn.BackpropagatableNode):
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'id'

    def get_arity(self):
        return 1

    def tune_bias(self) -> bool:
        return False

    def tune_weights(self) -> bool:
        return False

    def derivative(self, arg_no: int, x):
        return 1

    def eval_fn(self, args: dict = None):
        return self.children[0].eval()


class TestBackpropagation(unittest.TestCase):
    def setUp(self):
        self.error_fn = lambda a, y: 1/2 * (a - y)**2
        self.error_derivative = lambda a, y: (a - y)

    def test_root_exp(self):
        root = mn.Exp()
        inp = mn.Const(None)

        root.children = [inp]
        inp.parent = root
        inp.parent_index = 0

        values = [-np.e, -1, 0, 1, np.e]
        cases = [(x, a, b, y)
                 for x in values
                 for a in values
                 for b in values
                 for y in values]
        for x, a, b, y in cases:
            with self.subTest(x=x, a=a, b=b, y=y):
                root.bias = np.array([b], dtype=np.float64)
                root.weights = np.array([a], dtype=np.float64)
                inp.data = x
                inp.notify_child_changed(0)
                root.eval()
                mn.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    root.d_bias,
                    [np.exp(a * x + b) * (np.exp(a * x + b) - y)]
                )
                np.testing.assert_allclose(
                    root.d_weights,
                    [x * np.exp(a * x + b) * (np.exp(a * x + b) - y)]
                )

    def test_nonroot_exp(self):
        root = Identity()
        exp = mn.Exp()
        inp = mn.Const(None)

        root.children = [exp]
        exp.children = [inp]
        exp.parent = root
        exp.parent_index = 0
        inp.parent = exp
        inp.parent_index = 0

        values = [-np.e, -1, 0, 1, np.e]
        cases = [(x, a, b, y)
                 for x in values
                 for a in values
                 for b in values
                 for y in values]
        for x, a, b, y in cases:
            with self.subTest(x=x, a=a, b=b, y=y):
                exp.bias = np.array([b], dtype=np.float64)
                exp.weights = np.array([a], dtype=np.float64)
                inp.data = x
                inp.notify_child_changed(0)
                root.eval()
                mn.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    exp.d_bias,
                    [np.exp(a * x + b) * (np.exp(a * x + b) - y)]
                )
                np.testing.assert_allclose(
                    exp.d_weights,
                    [x * np.exp(a * x + b) * (np.exp(a * x + b) - y)]
                )

    def test_root_mul2(self):
        root = mn.Mul2()
        inp0 = mn.Const(None)
        inp1 = mn.Const(None)

        root.children = [inp0, inp1]
        inp0.parent = root
        inp0.parent_index = 0
        inp1.parent = root
        inp1.parent_index = 1

        values = [0, -1, 1]
        cases = [(x1, x2, a, b, y)
                 for x1 in values
                 for x2 in values
                 for a in values
                 for b in values
                 for y in values]
        for x1, x2, a, b, y in cases:
            with self.subTest(x1=x1, x2=x2, a=a, b=b, y=y):
                root.bias = np.array([a, b], dtype=np.float64)
                inp0.data = x1
                inp0.notify_child_changed(0)
                inp1.data = x2
                inp1.notify_child_changed(0)
                root.eval()
                mn.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    root.d_bias,
                    [(b + x2) * ((a + x1) * (b + x2) - y),
                     (a + x1) * ((a + x1) * (b + x2) - y)]
                )
                self.assertRaises(AttributeError, lambda: root.d_weights)


if __name__ == '__main__':
    unittest.main()
