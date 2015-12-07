# -*- coding: utf8 -*-
"""This module contains common mathematical functions and operators as
subclasses of :class:`evo.gp.support.GpNode` for use in symbolic regression
tasks.
"""

import evo.gp.support
import numpy


# noinspection PyAbstractClass
class MathNode(evo.gp.support.GpNode):
    """Base class for all mathematical nodes defined in
    :mod:`evo.gp.math_nodes`.
    """

    def __init__(self, cache=True):
        super().__init__()
        self.cache = cache
        self._cache = None

    def eval(self, args: dict=None):
        """Evaluates this node.

        If :attr:`.cache` is set to ``True`` and there is a cached value it is
        returned without the actual evaluation, otherwise the value is computed,
        stored to the cache and returned.

        .. note::

            You shouldn't override this method. Override :meth:`._eval` instead.

        :param args: values to set to variables, keyed by variable names
        :return: result of the evaluation
        """
        if not self.cache:
            return self.eval_fn(args)

        if self._cache is None:
            self._cache = self.eval_fn(args)

        return self._cache

    def eval_fn(self, args: dict=None):
        """Evaluation function of the node.

        Override this method to implement specific nodes. This method shouldn't
        be called directly. Use :meth:`.eval` instead, including calls for
        children's values.

        .. seealso:: :meth:`.eval`

        :param args: values to set to variables, keyed by variable names
        :return: result of the evaluation
        """
        raise NotImplementedError()

    def notify_child_changed(self, child_index: int, data=None):
        if self.cache:
            self._cache = None
            self.notify_change(data=data)

    def clone(self):
        c = super().clone()
        c.cache = self.cache
        c._cache = self._cache
        return c


# noinspection PyAbstractClass
class BackpropagatableNode(MathNode):
    def __init__(self, cache=True):
        super().__init__()
        self.cache = cache
        self._cache = None
        self.bias = numpy.zeros(self.get_arity())
        self.weights = numpy.ones(self.get_arity())
        self.argument = numpy.empty(self.get_arity())
        self.argument[:] = numpy.nan

    def clone(self):
        c = super().clone()
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias
        c.weights = self.weights.copy()
        return c

    def derivative(self, arg_no: int, x):
        """Returns the value of the derivative of the node's function, related
        to the given argument, at ``x``.

        :param arg_no: number of argument the derivative is related to, counted
            from 0
        :param x: point the derivative is to be computed at
        """
        raise NotImplementedError()

    def tune_bias(self) -> bool:
        """Returns whether the node's bias is subject to optimisation.
        """
        raise NotImplementedError()

    def tune_weights(self) -> bool:
        """Returns whether the node's weights are subject to optimisation.
        """
        raise NotImplementedError()


# noinspection PyAbstractClass
class WeightedOnlyNode(BackpropagatableNode):
    def tune_bias(self) -> bool:
        return False

    def tune_weights(self) -> bool:
        return True


# noinspection PyAbstractClass
class BiasedOnlyNode(BackpropagatableNode):
    def tune_bias(self) -> bool:
        return True

    def tune_weights(self) -> bool:
        return False


# noinspection PyAbstractClass
class BiasedWeightedNode(BackpropagatableNode):
    def tune_bias(self) -> bool:
        return True

    def tune_weights(self) -> bool:
        return True


class Add2(MathNode):
    """Addition of two operands: ``a + b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '+'
        self.bias = None

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        return a + b


class Sub2(MathNode):
    """Subtraction of the second operand from the first one: ``a - b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '-'

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        return a - b


class Mul2(BiasedOnlyNode):
    """Multiplication of two operands: ``a * b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '*'

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        self.argument[0] = self.children[0].eval(args) + self.bias[0]
        self.argument[1] = self.children[1].eval(args) + self.bias[1]
        return numpy.prod(self.argument)

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return x[1]
        if arg_no == 1:
            return x[0]
        raise ValueError('Invalid arg_no.')


class Div2(MathNode):
    """Division of the first operand by the second one: ``a / b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '/'

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        return numpy.true_divide(a, b)


class IDiv2(MathNode):
    """Integer division of the first operand by the second one: ``a // b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '//'

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        return numpy.floor_divide(a, b)


class PDiv2(MathNode):
    """Protected division of the first operand by the second one, returns 1 if
    the second operand is zero: ``1 if b == 0 else a / b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '{/}'

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        out = numpy.true_divide(a, b)
        if isinstance(b, numpy.ndarray):
            out[b == 0] = 1
            return out
        if b == 0:
            return 1
        return out


class PIDiv2(MathNode):
    """Protected integer division of the first operand by the second one,
    returns 1 if the second operand is zero: ``1 if b == 0 else a // b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '{//}'

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        out = numpy.floor_divide(a, b)
        if isinstance(b, numpy.ndarray):
            out[b == 0] = 1
            return out
        if b == 0:
            return 1
        return out


class Sin(BiasedWeightedNode):
    """The sine function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sin'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument[0] = self.weights[0] * a + self.bias
        return numpy.sin(self.argument)

    def derivative(self, arg_no: int, x):
        return numpy.cos(x)


class Cos(BiasedWeightedNode):
    """The cosine function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'cos'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights[0] * a + self.bias
        return numpy.cos(self.argument)

    def derivative(self, arg_no: int, x):
        return -numpy.sin(x)


class Exp(BiasedWeightedNode):
    """The natural exponential function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'exp'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights[0] * a + self.bias
        return numpy.exp(self.argument)

    def derivative(self, arg_no: int, x):
        return numpy.exp(x)


class Abs(MathNode):
    """Absolute value.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'abs'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.abs(a)


class Power(BiasedWeightedNode):
    """Power function, i.e. the argument raised to the power given in
    constructor.

    .. warning::

        This is an unprotected power, i.e. undefined powers (e.g. negative power
        of zero or half power of negative number) are not handled in this node.
    """
    def __init__(self, power=None, cache=True):
        super().__init__(cache)
        self.data = 'pow' + str(power)
        self.power = power

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights[0] * a + self.bias
        return numpy.power(self.argument, self.power)

    def clone(self):
        c = super().clone()
        c.power = self.power
        return c

    def derivative(self, arg_no: int, x):
        return self.power * numpy.power(x, self.power - 1)


class Sqrt(MathNode):
    """Square root.

    .. warning::

        This is an unprotected square root, i.e. the square root of negative
        numbers is not handled in this node.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sqrt'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.sqrt(a)


class PSqrt(MathNode):
    """Protected square root, returns the square root of the absolute value of
    the argument.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sqrt'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        a = numpy.abs(a)
        return numpy.sqrt(a)


class Sigmoid(BiasedWeightedNode):
    """Sigmoid function: :math:`sig(x) = \\frac{1}{1 + \\mathrm{e}^{-x}}`
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sig'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights[0] * a + self.bias
        return 1 / (1 + numpy.exp(-self.argument))

    def derivative(self, arg_no: int, x):
        a = 1 / (1 + numpy.exp(-x))
        return a * (1 - a)


class Sinc(BiasedWeightedNode):
    """The sinc function: :math:`sinc(x) = \\frac{\\sin{\\pi x}}{\\pi x}`,
    :math:`sinc(0) = 1`.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sinc'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights[0] * a + self.bias
        return numpy.sinc(self.argument)

    def derivative(self, arg_no: int, x):
        return (x * numpy.cos(x) - numpy.sin(x)) / x**2


class Softplus(BiasedWeightedNode):
    """The softplus or rectifier function:
    :math:`softplus(x) = \\ln(1 + \\mathrm{e}^{x})`.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'softplus'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights[0] * a + self.bias
        return numpy.log1p(numpy.exp(self.argument))

    def derivative(self, arg_no: int, x):
        a = numpy.exp(x)
        return a / (a + 1)


class Const(MathNode):
    """A constant.
    """

    def __init__(self, val=None, cache=True):
        super().__init__(cache)
        self.data = val

    def get_arity(self):
        return 0

    def eval_fn(self, args: dict=None):
        return self.data


class Variable(MathNode):
    """A variable.
    """

    def __init__(self, name=None, cache=True):
        super().__init__(cache)
        self.data = name

    def get_arity(self):
        return 0

    def eval_fn(self, args: dict=None):
        return args[self.data]


def backpropagate(root: BackpropagatableNode, cost_derivative: callable,
                  true_output, args):
    # root
    # bias derivative
    root.d_bias = numpy.empty_like(root.bias)
    for i in range(len(root.d_bias)):
        root.d_bias[i] = cost_derivative(root.eval(args), true_output) *\
                         root.derivative(i, root.argument)
    # weight derivative
    if root.tune_weights():
        inputs = list(map(lambda x: x.eval(args), root.children))
        root.d_weights = root.d_bias * numpy.array(inputs)

    # inner nodes
    o = list(root.children)
    while o:
        node = o.pop(0)
        if not isinstance(node, BackpropagatableNode):
            continue
        # bias derivative
        node.d_bias = numpy.empty_like(node.bias)
        for i in range(len(node.d_bias)):
            node.d_bias[i] = node.parent.d_bias[node.parent_index] *\
                             node.parent.weights[node.parent_index] *\
                             node.derivative(i, node.argument)

        if node.tune_weights():
            inputs = list(map(lambda x: x.eval(args), node.children))
            node.d_weights = node.d_bias * numpy.array(inputs)

        if not node.is_leaf():
            o.extend(node.children)
