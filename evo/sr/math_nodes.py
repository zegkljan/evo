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
            return self._eval(args)

        if self._cache is None:
            self._cache = self._eval(args)

        return self._cache

    def _eval(self, args: dict=None):
        """Evaluation function of the node.

        Override this method to implement specific nodes. This method shouldn't
        be called directly. Use :meth:`.eval` instead, including calls for
        children's values.

        .. seealso:: :meth:`.eval`

        :param args: values to set to variables, keyed by variable names
        :return: result of the evaluation
        """

    def notify_child_changed(self, child_index: int, data=None):
        if self.cache:
            self._cache = None
            self.notify_change(data=data)

    def clone(self):
        c = super().clone()
        c.cache = self.cache
        c._cache = self._cache
        return c


class Add2(MathNode):
    """Addition of two operands: ``a + b``
    """

    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '+'

    def get_arity(self):
        return 2

    def _eval(self, args: dict=None):
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

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        return a - b


class Mul2(MathNode):
    """Multiplication of two operands: ``a * b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '*'

    def get_arity(self):
        return 2

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        return a * b


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

    def _eval(self, args: dict=None):
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

    def _eval(self, args: dict=None):
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

    def _eval(self, args: dict=None):
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

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        out = numpy.floor_divide(a, b)
        if isinstance(b, numpy.ndarray):
            out[b == 0] = 1
            return out
        if b == 0:
            return 1
        return out


class Sin(MathNode):
    """The sine function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sin'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.sin(a)


class Cos(MathNode):
    """The cosine function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'cos'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.cos(a)


class Exp(MathNode):
    """The natural exponential function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'exp'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.exp(a)


class Abs(MathNode):
    """Absolute value.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'abs'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.abs(a)


class Power(MathNode):
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

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.power(a, self.power)

    def clone(self):
        c = super().clone()
        c.power = self.power
        return c


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

    def _eval(self, args: dict=None):
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

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        a = numpy.abs(a)
        return numpy.sqrt(a)


class Sigmoid(MathNode):
    """Sigmoid function: :math:`sig(x) = \\frac{1}{1 + \\mathrm{e}^{-x}}`
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sig'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return 1 / (1 + numpy.exp(-a))


class Sinc(MathNode):
    """The sinc function: :math:`sinc(x) = \\frac{\\sin{\\pi x}}{\\pi x}`,
    :math:`sinc(0) = 1`.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sig'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.sinc(a)


class Softplus(MathNode):
    """The softplus or rectifier function:
    :math:`softplus(x) = \\ln(1 + \\mathrm{e}^{x})`.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'softplus'

    def get_arity(self):
        return 1

    def _eval(self, args: dict=None):
        a = self.children[0].eval(args)
        return numpy.log1p(numpy.exp(a))


class Const(MathNode):
    """A constant.
    """

    def __init__(self, val=None, cache=True):
        super().__init__(cache)
        self.data = val

    def get_arity(self):
        return 0

    def _eval(self, args: dict=None):
        return self.data


class Variable(MathNode):
    """A variable.
    """

    def __init__(self, name=None, cache=True):
        super().__init__(cache)
        self.data = name

    def get_arity(self):
        return 0

    def _eval(self, args: dict=None):
        return args[self.data]
