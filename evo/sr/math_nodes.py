# -*- coding: utf8 -*-
"""This module contains common mathematical functions and operators as
subclasses of :class:`evo.gp.support.GpNode` for use in symbolic regression
tasks.
"""

import evo.gp.support
import numpy
import random


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

    def clear_cache(self, propagate: bool=True):
        """Clears the evaluation cache.

        If ``propagate`` is ``True`` (default) then this method will be
        recursively called on the children.
        """
        self._cache = None
        if not self.is_leaf() and propagate:
            for c in self.children:
                try:
                    c.clear_cache()
                except AttributeError:
                    pass

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
        self.argument = numpy.empty((self.get_arity(), 1))
        self.argument[:, 0] = numpy.nan

        """Specifies whether the bias in this node is subject to
        optimisation."""
        self.tune_bias = True
        """Specifies whether the weights in this node are subject to
        optimisation."""
        self.tune_weights = True

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

    def full_infix(self, num_format='.3f') -> str:
        """Returns the string representation of the tree in an infix form with
        all the weights and biases.

        :param num_format: format for weights and biases
        :return: the string representation of the tree
        """
        raise NotImplementedError()


class Add2(BackpropagatableNode):
    """Addition of two operands: ``a + b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '+'
        self.tune_bias = False

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        self.argument = numpy.empty((numpy.shape([a])[-1], 2))
        self.argument[:, 0] = a * self.weights[0]
        self.argument[:, 1] = b * self.weights[1]
        return numpy.sum(self.argument, axis=1)

    def derivative(self, arg_no: int, x):
        return 1

    def full_infix(self, num_format='.3f'):
        return ('({0:' + num_format + '} * {2} + '
                '{1:' + num_format + '} * {3})').format(
            self.weights[0], self.weights[1],
            self.children[0].full_infix(num_format),
            self.children[1].full_infix(num_format))


class Sub2(BackpropagatableNode):
    """Subtraction of the second operand from the first one: ``a - b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '-'
        self.tune_bias = False

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        self.argument = numpy.empty((numpy.shape([a])[-1], 2))
        self.argument[:, 0] = a * self.weights[0]
        self.argument[:, 1] = -b * self.weights[1]
        return numpy.sum(self.argument, axis=1)

    def derivative(self, arg_no: int, x):
        return 1

    def full_infix(self, num_format='.3f'):
        return ('({0:' + num_format + '} * {2} - '
                '{1:' + num_format + '} * {3})').format(
            self.weights[0], self.weights[1],
            self.children[0].full_infix(num_format),
            self.children[1].full_infix(num_format))


class Mul2(BackpropagatableNode):
    """Multiplication of two operands: ``a * b``
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '*'
        self.tune_weights = False

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        self.argument = numpy.empty((numpy.shape([a])[-1], 2))
        self.argument[:, 0] = a + self.bias[0]
        self.argument[:, 1] = b + self.bias[1]
        return numpy.prod(self.argument, axis=1)

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return x[:, 1]
        if arg_no == 1:
            return x[:, 0]
        raise ValueError('Invalid arg_no.')

    def full_infix(self, num_format='.3f'):
        return ('(({0:' + num_format + '} + {2}) * '
                '({1:' + num_format + '} + {3}))').format(
            self.bias[0], self.bias[1],
            self.children[0].full_infix(num_format),
            self.children[1].full_infix(num_format))


class Div2(BackpropagatableNode):
    """Division of the first operand by the second one: ``a / b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = '/'
        self.tune_weights = False

    def get_arity(self):
        return 2

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        b = self.children[1].eval(args)
        self.argument = numpy.empty((a.shape[0], 2))
        self.argument[:, 0] = a + self.bias[0]
        self.argument[:, 1] = b + self.bias[1]
        return numpy.true_divide(self.argument[0], self.argument[1])

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return 1.0 / x[1]
        if arg_no == 1:
            return -numpy.true_divide(x[0], numpy.square(x[1]))
        raise ValueError('Invalid arg_no.')

    def full_infix(self, num_format='.3f'):
        return ('(({0:' + num_format + '} + {2}) / '
                '({1:' + num_format + '} + {3}))').format(
            self.bias[0], self.bias[1],
            self.children[0].full_infix(num_format),
            self.children[1].full_infix(num_format))


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


class Sin(BackpropagatableNode):
    """The sine function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sin'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights * a + self.bias
        return numpy.sin(self.argument[0])

    def derivative(self, arg_no: int, x):
        return numpy.cos(x)

    def full_infix(self, num_format='.3f'):
        return ('sin({0:' + num_format + '} + '
                '{1:' + num_format + '} * {3})').format(
            self.bias[0], self.weights[1],
            self.children[0].full_infix(num_format))


class Cos(BackpropagatableNode):
    """The cosine function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'cos'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights * a + self.bias
        return numpy.cos(self.argument[0])

    def derivative(self, arg_no: int, x):
        return -numpy.sin(x)

    def full_infix(self, num_format='.3f'):
        return ('cos({0:' + num_format + '} + '
                '{1:' + num_format + '} * {3})').format(
            self.bias[0], self.weights[1],
            self.children[0].full_infix(num_format))


class Exp(BackpropagatableNode):
    """The natural exponential function.
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'exp'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights * a + self.bias
        return numpy.exp(self.argument)

    def derivative(self, arg_no: int, x):
        return numpy.exp(x)

    def full_infix(self, num_format='.3f'):
        return ('exp({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].full_infix(num_format))


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


class Power(BackpropagatableNode):
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
        self.argument = self.weights * a + self.bias
        return numpy.power(self.argument[0], self.power)

    def clone(self):
        c = super().clone()
        c.power = self.power
        return c

    def derivative(self, arg_no: int, x):
        return self.power * numpy.power(x[0], self.power - 1)

    def full_infix(self, num_format='.3f'):
        return ('(({0:' + num_format + '} + '
                '{1:' + num_format + '} * {4})^{3})').format(
            self.bias[0], self.weights[0], self.power,
            self.children[0].full_infix(num_format))


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


class Sigmoid(BackpropagatableNode):
    """Sigmoid function: :math:`sig(x) = \\frac{1}{1 + \\mathrm{e}^{-x}}`
    """
    def __init__(self, cache=True):
        super().__init__(cache)
        self.data = 'sig'

    def get_arity(self):
        return 1

    def eval_fn(self, args: dict=None):
        a = self.children[0].eval(args)
        self.argument = self.weights * a + self.bias
        return 1 / (1 + numpy.exp(-self.argument[0]))

    def derivative(self, arg_no: int, x):
        a = 1 / (1 + numpy.exp(-x[0]))
        return a * (1 - a)

    def full_infix(self, num_format='.3f'):
        return ('sig({0:' + num_format + '} + '
                '{1:' + num_format + '} * {3})').format(
            self.bias[0], self.weights[0],
            self.children[0].full_infix(num_format))


class Sinc(BackpropagatableNode):
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
        self.argument = self.weights * a + self.bias
        return numpy.sinc(self.argument[0])

    def derivative(self, arg_no: int, x):
        return (x * numpy.cos(x[0]) - numpy.sin(x[0])) / x**2

    def full_infix(self, num_format='.3f'):
        return ('sinc({0:' + num_format + '} + '
                '{1:' + num_format + '} * {3})').format(
            self.bias[0], self.weights[0],
            self.children[0].full_infix(num_format))


class Softplus(BackpropagatableNode):
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
        self.argument = self.weights * a + self.bias
        return numpy.log1p(numpy.exp(self.argument[0]))

    def derivative(self, arg_no: int, x):
        a = numpy.exp(x[0])
        return a / (a + 1)

    def full_infix(self, num_format='.3f'):
        return ('softplus({0:' + num_format + '} + '
                '{1:' + num_format + '} * {3})').format(
            self.bias[0], self.weights[0],
            self.children[0].full_infix(num_format))


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

    def full_infix(self, num_format='.3f'):
        return ('{' + num_format + '}').format(self.data)


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

    def full_infix(self, num_format='.3f'):
        return self.data


def backpropagate(root: BackpropagatableNode, cost_derivative: callable,
                  true_output, args, datapts_no=1):
    """Computes the gradient of the cost function in weights and biases using
    the back-propagation algorithm.

    Computes the partial derivatives of the error (cost) function with respect
    to the weights and biases of the given tree.

    The partial derivatives are stored in the nodes in the ``d_bias`` and
    ``d_weights`` attributes. If node's
    :attr:`BackpropagatableNode.tune_weights` is set to ``False`` then the
    weight partial derivatives are not computed at all. On the other hand,
    bias partial derivative is always computed because it is needed for the
    computation of the children's values.

    The back-propagation runs from the root to the leaves. However, if a node
    that is not an instance of :class:`BackpropagatableNode` is encountered the
    propagation through this node is not performed, hence cutting its subtree
    off the algorithm.

    The ``cost_function`` represents the derivative of the error function with
    respect to the tree's output. It is supposed to be a callable of two
    arguments, the first argument being the tree's output, the second argument
    being the true output.

    The ``datapts_no`` argument is used to tell the backprop. algorithm how many
    datapoints the propagation is computed for. Defaults to 1. This number must
    match with the actual number of datapoints otherwise there will be errors.

    .. note::

        This function does not perform any learning. It merely computes the
        gradient.

    :param root: root of the tree that is subject to optimisation
    :param cost_derivative: derivative of the error function
    :param true_output: the desired output
    :param args: arguments for evaluation
    """
    # root
    # bias derivative
    root.d_bias = numpy.empty((datapts_no, len(root.bias)))
    for i in range(len(root.bias)):
        root.d_bias[:, i] = cost_derivative(root.eval(args), true_output) *\
                            root.derivative(i, root.argument)
    # weight derivative
    if root.tune_weights:
        inputs = numpy.array([x.eval(args) for x in root.children])
        root.d_weights = root.d_bias * inputs.T

    # inner nodes
    o = list(root.children)
    while o:
        node = o.pop(0)
        if not isinstance(node, BackpropagatableNode):
            continue
        # bias derivative
        node.d_bias = numpy.empty((datapts_no, len(node.bias)))
        for i in range(len(node.bias)):
            node.d_bias[:, i] = node.parent.d_bias[:, node.parent_index] *\
                                node.parent.weights[node.parent_index] *\
                                node.derivative(i, node.argument)

        if node.tune_weights:
            inputs = numpy.array([x.eval(args) for x in node.children])
            node.d_weights = node.d_bias * inputs.T

        if not node.is_leaf():
            o.extend(node.children)


def sgd_step(root: BackpropagatableNode, train_inputs, train_output,
             cost_derivative: callable, var_mapping: dict, minibatch_size=None,
             eta=0.01, generator: random.Random=None):
    """Performs one step (epoch) of a Stochastic Gradient Descent algorithm.

    Performs one step of the SGD algorithm by creating a "minibatch" (of size
    ``minibatch_size``) from the supplied training data (``train_inputs`` and
    ``train_output``), evaluating it by the tree, computing the gradient using
    :func:`backpropagate` and updating the weights and biases based on the
    gradient and the learning rate ``eta``.

    The ``var_mapping`` argument is responsible for mapping the input variables
    to variable names of the tree. It is supposed to be a dict with keys being
    integers counting from 0 and values being variable names in the tree. The
    keys are supposed to correspond to the column indices to ``train_inputs``.

    .. note::

        The training data arrays will be shuffled in order to randomly assemble
        the minibatch. If the arrays need to be unchanged, copy them beforehand.

    :param root: root of the tree to be updated
    :param train_inputs: training inputs; one row is expected to be one
        datapoint
    :param train_output: training outputs; one row is expected to be one
        datapoint
    :param cost_derivative: derivative of the error function, see
        :func:`backpropagate`
    :param minibatch_size: number of datapoints in the minibatch
    :param eta: learning rate, default is 0.01
    """
    if generator is None:
        generator = random

    if minibatch_size == train_inputs.shape[0]:
        mb_input = train_inputs
        mb_output = train_output
    else:
        mb_indices = generator.sample(range(train_inputs.shape[0]), minibatch_size)
        mb_input = train_inputs[mb_indices, :]
        mb_output = train_output[mb_indices]

    args = {var_mapping[num]: mb_input[:, num] for num in var_mapping}

    root.clear_cache()
    backpropagate(root, cost_derivative, mb_output, args, minibatch_size)

    factor = eta / minibatch_size

    def update(node: BackpropagatableNode):
        if not isinstance(node, BackpropagatableNode):
            return
        if node.tune_bias:
            d_bias = numpy.sum(node.d_bias, axis=0)
            node.bias = node.bias - factor * d_bias

        if node.tune_weights:
            d_weights = numpy.sum(node.d_weights, axis=0)
            node.weights = node.weights - factor * d_weights

    root.preorder(update)


def rprop_step(root: BackpropagatableNode, train_inputs, train_output,
               cost_derivative: callable, var_mapping: dict, delta_init=0.1,
               delta_min=1e-6, delta_max=50, eta_minus=0.5, eta_plus=1.2):
    """Performs one step (epoch) of the Rprop (**R**esilient
    back**prop**agation) algorithm.

    The ``var_mapping`` argument is responsible for mapping the input variables
    to variable names of the tree. It is supposed to be a dict with keys being
    integers counting from 0 and values being variable names in the tree. The
    keys are supposed to correspond to the column indices to ``train_inputs``.

    .. note::

        The training data arrays will be shuffled in order to randomly assemble
        the minibatch. If the arrays need to be unchanged, copy them beforehand.

    :param root: root of the tree to be updated
    :param train_inputs: training inputs; one row is expected to be one
        datapoint
    :param train_output: training outputs; one row is expected to be one
        datapoint
    :param cost_derivative: derivative of the error function, see
        :func:`backpropagate`
    """
    args = {var_mapping[num]: train_inputs[:, num] for num in var_mapping}

    root.clear_cache()
    backpropagate(root, cost_derivative, train_output, args,
                  train_inputs.shape[0])

    def update(node: BackpropagatableNode):
        if not isinstance(node, BackpropagatableNode):
            return
        if node.tune_bias:
            d_bias = numpy.sum(node.d_bias, axis=0)
            if not hasattr(node, 'prev_d_bias'):
                node.prev_d_bias = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_bias'):
                node.delta_bias = numpy.ones(node.bias.shape) * delta_init

            s_prev = numpy.sign(node.prev_d_bias)
            s = numpy.sign(d_bias)
            for i in range(node.get_arity()):
                if s_prev[i] * s[i] > 0:
                    node.delta_bias[i] *= eta_plus
                    if node.delta_bias[i] > delta_max:
                        node.delta_bias[i] = delta_max
                    node.bias[i] -= s[i] * node.delta_bias[i]
                    node.prev_d_bias[i] = d_bias[i]
                elif s_prev[i] * s[i] < 0:
                    node.delta_bias[i] *= eta_minus
                    if node.delta_bias[i] < delta_min:
                        node.delta_bias[i] = delta_min
                    node.prev_d_bias[i] = 0
                else:
                    node.bias[i] -= node.delta_bias[i] * s[i]
                    node.prev_d_bias[i] = d_bias[i]

        if node.tune_weights:
            d_weights = numpy.sum(node.d_weights, axis=0)
            if not hasattr(node, 'prev_d_weights'):
                node.prev_d_weights = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_weights'):
                node.delta_weights = numpy.ones(node.weights.shape) * delta_init

            s_prev = numpy.sign(node.prev_d_weights)
            s = numpy.sign(d_weights)
            for i in range(node.get_arity()):
                if s_prev[i] * s[i] > 0:
                    node.delta_weights[i] *= eta_plus
                    if node.delta_weights[i] > delta_max:
                        node.delta_weights[i] = delta_max
                    node.weights[i] -= s[i] * node.delta_weights[i]
                    node.prev_d_weights[i] = d_weights[i]
                elif s_prev[i] * s[i] < 0:
                    node.delta_weights[i] *= eta_minus
                    if node.delta_weights[i] < delta_min:
                        node.delta_weights[i] = delta_min
                    node.prev_d_weights[i] = 0
                else:
                    node.weights[i] -= node.delta_weights[i] * s[i]
                    node.prev_d_weights[i] = d_weights[i]

    root.preorder(update)
