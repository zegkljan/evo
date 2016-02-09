# -*- coding: utf8 -*-

import unittest

import numpy as np

import evo.sr.backpropagation
import evo.sr.math_nodes as mn


class Identity(evo.sr.backpropagation.WeightedNode):
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

        values = [0, -1, 1, -np.e, np.e]
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
                inp.child_changed(0)
                root.eval()
                evo.sr.backpropagation.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    root.d_bias,
                    [[np.exp(a * x + b) * (np.exp(a * x + b) - y)]]
                )
                np.testing.assert_allclose(
                    root.d_weights,
                    [[x * np.exp(a * x + b) * (np.exp(a * x + b) - y)]]
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

        values = [0, -1, 1, -np.e, np.e]
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
                inp.child_changed(0)
                root.eval()
                evo.sr.backpropagation.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    exp.d_bias,
                    [[np.exp(a * x + b) * (np.exp(a * x + b) - y)]]
                )
                np.testing.assert_allclose(
                    exp.d_weights,
                    [[x * np.exp(a * x + b) * (np.exp(a * x + b) - y)]]
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

        values = [0, -0.5, 0.5, -1.5, 1.5]
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
                inp0.child_changed(0)
                inp1.data = x2
                inp1.child_changed(0)
                root.eval()
                evo.sr.backpropagation.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    root.d_bias,
                    [[(b + x2) * ((a + x1) * (b + x2) - y),
                      (a + x1) * ((a + x1) * (b + x2) - y)]]
                )
                self.assertRaises(AttributeError, lambda: root.d_weights)

    def test_nonroot_mul2(self):
        root = Identity()
        mul = mn.Mul2()
        inp0 = mn.Const(None)
        inp1 = mn.Const(None)

        root.children = [mul]
        mul.parent = root
        mul.parent_index = 0
        mul.children = [inp0, inp1]
        inp0.parent = mul
        inp0.parent_index = 0
        inp1.parent = mul
        inp1.parent_index = 1

        values = [0, -0.5, 0.5, -1.5, 1.5]
        cases = [(x1, x2, a, b, y)
                 for x1 in values
                 for x2 in values
                 for a in values
                 for b in values
                 for y in values]
        for x1, x2, a, b, y in cases:
            with self.subTest(x1=x1, x2=x2, a=a, b=b, y=y):
                mul.bias = np.array([a, b], dtype=np.float64)
                inp0.data = x1
                inp0.child_changed(0)
                inp1.data = x2
                inp1.child_changed(0)
                root.eval()
                evo.sr.backpropagation.backpropagate(root, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    mul.d_bias,
                    [[(b + x2) * ((a + x1) * (b + x2) - y),
                      (a + x1) * ((a + x1) * (b + x2) - y)]]
                )
                self.assertRaises(AttributeError, lambda: mul.d_weights)

    def test_exp_mul2(self):
        exp = mn.Exp()
        mul = mn.Mul2()
        inp0 = mn.Const(None)
        inp1 = mn.Const(None)

        exp.children = [mul]
        mul.parent = exp
        mul.parent_index = 0
        mul.children = [inp0, inp1]
        inp0.parent = mul
        inp0.parent_index = 0
        inp1.parent = mul
        inp1.parent_index = 1

        values = [0, -1.5, 1.5]
        cases = [(x1, x2, a, b, c, d, y)
                 for x1 in values  # input 1
                 for x2 in values  # input 2
                 for a in values   # multiplication bias 1
                 for b in values   # multiplication bias 2
                 for c in values   # exp bias
                 for d in values   # exp weight
                 for y in values]  #
        for x1, x2, a, b, c, d, y in cases:
            with self.subTest(x1=x1, x2=x2, a=a, b=b, c=c, d=d, y=y):
                mul.bias = np.array([a, b], dtype=np.float64)
                exp.bias = np.array([c], dtype=np.float64)
                exp.weights = np.array([d], dtype=np.float64)
                inp0.data = x1
                inp0.child_changed(0)
                inp1.data = x2
                inp1.child_changed(0)
                exp.eval()
                evo.sr.backpropagation.backpropagate(exp, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    mul.d_bias,
                    [[d * (b + x2) * np.exp(d * (a + x1) * (b + x2) + c) * (np.exp(d * (a + x1) * (b + x2) + c) - y),
                      d * (a + x1) * np.exp(d * (a + x1) * (b + x2) + c) * (np.exp(d * (a + x1) * (b + x2) + c) - y)]]
                )
                self.assertRaises(AttributeError, lambda: mul.d_weights)
                np.testing.assert_allclose(
                    exp.d_bias,
                    [[np.exp(d * (a + x1) * (b + x2) + c) * (np.exp(d * (a + x1) * (b + x2) + c) - y)]]
                )
                np.testing.assert_allclose(
                    exp.d_weights,
                    [[(a + x1) * (b + x2) * np.exp(d * (a + x1) * (b + x2) + c) * (np.exp(d * (a + x1) * (b + x2) + c) - y)]]
                )

    def test_mul2_exp(self):
        mul = mn.Mul2()
        exp = mn.Exp()
        ident = Identity()
        inp0 = mn.Const(None)
        inp1 = mn.Const(None)

        mul.children = [exp, ident]
        exp.parent = mul
        exp.parent_index = 0
        exp.children = [inp0]
        inp0.parent = exp
        inp0.parent_index = 0
        ident.parent = mul
        ident.parent_index = 1
        ident.children = [inp1]
        inp1.parent = ident
        inp1.parent_index = 0

        values = [0, -1.5, 1.5]
        cases = [(x1, x2, a, b, c, d, y)
                 for x1 in values  # input 1
                 for x2 in values  # input 2
                 for a in values   # exp bias
                 for b in values   # exp weight
                 for c in values   # multiplication bias 1
                 for d in values   # multiplication bias 2
                 for y in values]  #
        for x1, x2, a, b, c, d, y in cases:
            with self.subTest(x1=x1, x2=x2, a=a, b=b, c=c, d=d, y=y):
                exp.bias = np.array([a], dtype=np.float64)
                exp.weights = np.array([b], dtype=np.float64)
                mul.bias = np.array([c, d], dtype=np.float64)
                inp0.data = x1
                inp0.child_changed(0)
                inp1.data = x2
                inp1.child_changed(0)
                mul.eval()
                evo.sr.backpropagation.backpropagate(mul, self.error_derivative, y, {})
                np.testing.assert_allclose(
                    exp.d_bias,
                    [[(d + x2) * np.exp(a + b * x1) * ((d + x2) * (np.exp(a + b * x1) + c) - y)]]
                )
                np.testing.assert_allclose(
                    exp.d_weights,
                    [[x1 * (d + x2) * np.exp(a + b * x1) * ((d + x2) * (np.exp(a + b * x1) + c) - y)]]
                )
                np.testing.assert_allclose(
                    mul.d_bias,
                    [[(d + x2) * ((d + x2) * (np.exp(a + b * x1) + c) - y),
                      (np.exp(a + b * x1) + c) * ((d + x2) * (np.exp(a + b * x1) + c) - y)]]
                )
                self.assertRaises(AttributeError, lambda: mul.d_weights)


if __name__ == '__main__':
    unittest.main()
