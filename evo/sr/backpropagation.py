"""This module contains algorithms and data structures for backpropagation-based
learning of tree expressions.
"""

import numpy
import random

import evo.sr


class WeightedNode(evo.sr.MathNode):
    """An abstract class for nodes that have a weight and/or bias assigned for
    their children, effectively carrying constants all along.
    """
    def __init__(self, tune_bias=True, tune_weights=True, **kwargs):
        """
        ``tune_bias`` specifies whether the bias in this node is subject to
        optimisation. Plain boolean value (e.g. ``True``\ ) specifies this for
        all biases. List of boolean values (e.g. ``[True, False``\ ) specifies
        this per argument on the corresponding positions.

        ``tune_weights`` works in exactly the same way as ``tune_bias``
        described above.

        :param cache: specifies whether the output of the node is going to be
            cached
        :type tune_bias: :class:`bool` or :class:`list` of bools
        :param tune_bias: specifies whether or which biases are allowed to be
            tuned
        :type tune_weights: :class:`bool` or :class:`list` of bools
        :param tune_weights: specifies whether or which weights are allowed to
            be tuned
        """
        super().__init__(**kwargs)
        self.bias = numpy.zeros(self.get_arity())
        self.weights = numpy.ones(self.get_arity())
        self.argument = None

        self.tune_bias = tune_bias
        self.tune_weights = tune_weights

    def eval_child(self, child_no: int, args: dict = None):
        val = super().eval_child(child_no, args)
        return self.weights[child_no] * val + self.bias[child_no]

    def operation(self, *args):
        l = max(map(lambda a: numpy.shape([a])[-1], args))
        self.argument = numpy.empty((l, self.get_arity()))
        for i in range(self.get_arity()):
            self.argument[:, i] = args[i]
        return super().operation(*args)

    def derivative(self, arg_no: int, x):
        """Returns the value of the derivative of the node's function, related
        to the given argument, at ``x``.

        :param arg_no: number of argument the derivative is related to, counted
            from 0
        :param x: point the derivative is to be computed at
        """
        raise NotImplementedError()

    def infix(self, **kwargs) -> str:
        return self.full_infix(**kwargs)

    def full_infix(self, **kwargs) -> str:
        """Returns the string representation of the tree in an infix form with
        all the weights and biases.

        :keyword num_format: format for weights and biases
        :return: the string representation of the tree
        """
        raise NotImplementedError()


class Add2(WeightedNode, evo.sr.Add2):
    """Weighted version of :class:`evo.sr.Add2`\ .

    .. seealso:: :class:`evo.sr.Add2`
    """
    def __init__(self, **kwargs):
        """Add2 does never use bias (only weights), therefore ``tune_bias``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_bias'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Add2, self).eval_child(child_no, args)
        return self.weights[child_no] * val

    def derivative(self, arg_no: int, x):
        return 1

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('({0:' + num_format + '} * {2} + '
                '{1:' + num_format + '} * {3})').format(
            self.weights[0], self.weights[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Add2(cache=self.cache, tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Sub2(WeightedNode, evo.sr.Sub2):
    """Weighted version of :class:`evo.sr.Sub2`\ .

    .. seealso:: :class:`evo.sr.Sub2`
    """
    def __init__(self, **kwargs):
        """Sub2 does never use bias (only weights), therefore ``tune_bias``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_bias'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Sub2, self).eval_child(child_no, args)
        return self.weights[child_no] * val

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return 1
        elif arg_no == 1:
            return -1
        raise ValueError('Invalid argument number.')

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('({0:' + num_format + '} * {2} - '
                '{1:' + num_format + '} * {3})').format(
            self.weights[0], self.weights[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Sub2(cache=self.cache, tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Mul2(WeightedNode, evo.sr.Mul2):
    """Weighted version of :class:`evo.sr.Mul2`\ .

    .. seealso:: :class:`evo.sr.Mul2`
    """
    def __init__(self, **kwargs):
        """Mul2 does never use weights (only biases), therefore ``tune_weights``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_weights'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Mul2, self).eval_child(child_no, args)
        return self.bias[child_no] + val

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return x[:, 1]
        if arg_no == 1:
            return x[:, 0]
        raise ValueError('Invalid arg_no.')

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('(({0:' + num_format + '} + {2}) * '
                '({1:' + num_format + '} + {3}))').format(
            self.bias[0], self.bias[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Mul2(cache=self.cache, tune_bias=self.tune_bias)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Div2(WeightedNode, evo.sr.Div2):
    """Weighted version of :class:`evo.sr.Div2`\ .

    .. seealso:: :class:`evo.sr.Div2`
    """
    def __init__(self, **kwargs):
        """Div2 does never use weights (only biases), therefore ``tune_weights``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_weights'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Div2, self).eval_child(child_no, args)
        return self.bias[child_no] + val

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return numpy.true_divide(1.0, numpy.square(x[:, 1]))
        if arg_no == 1:
            return -numpy.true_divide(x[:, 0], numpy.square(x[:, 1]))
        raise ValueError('Invalid arg_no.')

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('(({0:' + num_format + '} + {2}) / '
                '({1:' + num_format + '} + {3}))').format(
            self.bias[0], self.bias[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Div2(cache=self.cache, tune_bias=self.tune_bias)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Sin(WeightedNode, evo.sr.Sin):
    """Weighted version of :class:`evo.sr.Sin`\ .

    .. seealso:: :class:`evo.sr.Sin`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.cos(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('sin({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sin(cache=self.cache, tune_bias=self.tune_bias,
                tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Cos(WeightedNode, evo.sr.Cos):
    """Weighted version of :class:`evo.sr.Cos`\ .

    .. seealso:: :class:`evo.sr.Cos`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return -numpy.sin(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('cos({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Cos(cache=self.cache, tune_bias=self.tune_bias,
                tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Exp(WeightedNode, evo.sr.Exp):
    """Weighted version of :class:`evo.sr.Exp`\ .

    .. seealso:: :class:`evo.sr.Exp`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.exp(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('exp({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Exp(cache=self.cache, tune_bias=self.tune_bias,
                tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Abs(WeightedNode, evo.sr.Abs):
    """Weighted version of :class:`evo.sr.Abs`\ .

    .. seealso:: :class:`evo.sr.Abs`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.sign(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('|{0:' + num_format + '} + '
                '{1:' + num_format + '} * {2}|').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Abs(cache=self.cache, tune_bias=self.tune_bias,
                tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Power(WeightedNode, evo.sr.Power):
    """Weighted version of :class:`evo.sr.Power`\ .

    .. seealso:: :class:`evo.sr.Power`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return self.power * numpy.power(x[:, 0], self.power - 1)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        base = '(({0:' + num_format + '} + {1:' + num_format + '} * {3})^{2})'
        return base.format(
            self.bias[0], self.weights[0], self.power,
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Power(cache=self.cache, power=self.power, tune_bias=self.tune_bias,
                  tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Sigmoid(WeightedNode, evo.sr.Sigmoid):
    """Weighted version of :class:`evo.sr.Sigmoid`\ .

    .. seealso:: :class:`evo.sr.Sigmoid`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        a = 1 / (1 + numpy.exp(-x[:, 0]))
        return a * (1 - a)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('sig({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sigmoid(cache=self.cache, tune_bias=self.tune_bias,
                    tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Sinc(WeightedNode, evo.sr.Sinc):
    """Weighted version of :class:`evo.sr.Sinc`\ .

    .. seealso:: :class:`evo.sr.Sinc`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.true_divide(
            x[:, 0] * numpy.cos(x[:, 0]) - numpy.sin(x[:, 0]),
            x[:, 0]**2)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('sinc({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sinc(cache=self.cache, tune_bias=self.tune_bias,
                 tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


class Softplus(WeightedNode, evo.sr.Softplus):
    """Weighted version of :class:`evo.sr.Softplus`\ .

    .. seealso:: :class:`evo.sr.Softplus`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        a = numpy.exp(x[:, 0])
        return a / (a + 1)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        return ('softplus({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sigmoid(cache=self.cache, tune_bias=self.tune_bias,
                    tune_weights=self.tune_weights)
        c.cache = self.cache
        c._cache = self._cache
        c.bias = self.bias.copy()
        c.weights = self.weights.copy()
        if self.argument is not None:
            c.argument = self.argument.copy()
        return c


def backpropagate(root: WeightedNode, cost_derivative: callable,
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
    o = [root]
    while o:
        node = o.pop(0)
        if not isinstance(node, WeightedNode):
            continue
        # bias derivative
        # if root, do it different
        if node is root:
            node.d_bias = numpy.empty((datapts_no, node.bias.size))
            for i in range(node.bias.size):
                cd = cost_derivative(node.eval(args), true_output)
                nd = node.derivative(i, node.argument)
                node.d_bias[:, i] = cd * nd
        else:
            node.d_bias = numpy.empty((datapts_no, len(node.bias)))
            for i in range(len(node.bias)):
                pd = node.parent.d_bias[:, node.parent_index]
                pw = node.parent.weights[node.parent_index]
                nd = node.derivative(i, node.argument)
                node.d_bias[:, i] = pd * pw * nd

        # weights derivative
        inputs = None
        if node.tune_weights is True:
            raw = [x.eval(args) for x in node.children]
            if len(raw) == 1:
                inputs = numpy.array(raw)
            else:
                inputs = numpy.column_stack(numpy.broadcast(*raw))
        elif node.tune_weights:
            # noinspection PyUnresolvedReferences
            raw = [node.children[i].eval(args) if node.tune_weights[i] else 0
                   for i
                   in range(len(node.children))]
            inputs = numpy.column_stack(numpy.broadcast(*raw))
        if inputs is not None and len(inputs) > 0:
            node.d_weights = node.d_bias * inputs.T

        if not node.is_leaf():
            o.extend(node.children)


def sgd_step(root: WeightedNode, train_inputs, train_output,
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

    def update(node: WeightedNode):
        if not isinstance(node, WeightedNode):
            return
        if node.tune_bias:
            d_bias = numpy.sum(node.d_bias, axis=0)
            node.bias = node.bias - factor * d_bias

        if node.tune_weights:
            d_weights = numpy.sum(node.d_weights, axis=0)
            node.weights = node.weights - factor * d_weights

    root.preorder(update)


class RpropBase(object):
    """This is a base class for Rprop algorithm variants.
    """
    def __init__(self, delta_init=0.1, delta_min=1e-6, delta_max=50,
                 eta_minus=0.5, eta_plus=1.2):
        super().__init__()
        self.delta_init = delta_init
        self.eta_plus = eta_plus
        self.delta_max = delta_max
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.updated = False

    def update(self, root: WeightedNode, error, prev_error):
        self.updated = False
        root.preorder(self.update_node)
        return self.updated



    def update_node(self, node):
        raise NotImplementedError()


class RpropPlus(RpropBase):
    """This is the original Rprop algorithm [Riedmiller1993] (with
    weight-backtracking), also called Rprop\ :sup:`+` [Igel2000].

    .. [Riedmiller1993] Martin Riedmiller and Heinrich Braun. `A Direct Adaptive
        Method for Faster Backpropagation Learning: The RPROP Algorithm .`
        IEEE International Conference on Neural Networks. 1993. pp. 586-591.

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """

    def upd(self, w, d, prev_d, delta, prev_update, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
                w[i] -= prev_update[i]
                prev_d[i] = 0
            else:
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and hasattr(node, 'd_bias'):
            upd = True
            d_bias = numpy.sum(node.d_bias, axis=0)
            if not hasattr(node, 'prev_d_bias'):
                node.prev_d_bias = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_bias'):
                node.delta_bias = numpy.ones(node.bias.shape) * self.delta_init

            if not hasattr(node, 'prev_bias_update'):
                node.prev_bias_update = numpy.zeros(node.bias.shape)

            self.upd(node.bias, d_bias, node.prev_d_bias, node.delta_bias,
                     node.prev_bias_update, node.get_arity())

        if node.tune_weights and hasattr(node, 'd_weights'):
            upd = True
            d_weights = numpy.sum(node.d_weights, axis=0)
            if not hasattr(node, 'prev_d_weights'):
                node.prev_d_weights = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_weights'):
                node.delta_weights = numpy.ones(node.weights.shape) *\
                                     self.delta_init

            if not hasattr(node, 'prev_weight_update'):
                node.prev_weight_update = numpy.zeros(node.weights.shape)

            self.upd(node.weights, d_weights, node.prev_d_weights,
                     node.delta_weights, node.prev_weight_update,
                     node.get_arity())

        if upd:
            self.updated = True
            node.notify_change()


class RpropMinus(RpropBase):
    """This is the Rprop algorithm without weight-backtracking, also called
    Rprop\ :sup:`-` [Igel2000].

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """

    def upd(self, w, d, prev_d, delta, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
            w[i] -= s[i] * delta[i]
            prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and hasattr(node, 'd_bias'):
            upd = True
            d_bias = numpy.sum(node.d_bias, axis=0)
            if not hasattr(node, 'prev_d_bias'):
                node.prev_d_bias = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_bias'):
                node.delta_bias = numpy.ones(node.bias.shape) * self.delta_init

            self.upd(node.bias, d_bias, node.prev_d_bias, node.delta_bias,
                     node.get_arity())

        if node.tune_weights and hasattr(node, 'd_weights'):
            upd = True
            d_weights = numpy.sum(node.d_weights, axis=0)
            if not hasattr(node, 'prev_d_weights'):
                node.prev_d_weights = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_weights'):
                node.delta_weights = numpy.ones(node.weights.shape) *\
                                     self.delta_init

            self.upd(node.weights, d_weights, node.prev_d_weights,
                     node.delta_weights, node.get_arity())

        if upd:
            self.updated = True
            node.notify_change()


class IRpropPlus(RpropBase):
    """This is the improved Rprop algorithm with weight-backtracking, also called
    iRprop\ :sup:`+` [Igel2000].

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_worsened = False

    def upd(self, w, d, prev_d, delta, prev_update, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
                if self.error_worsened:
                    w[i] -= prev_update[i]
                prev_d[i] = 0
            else:
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and hasattr(node, 'd_bias'):
            upd = True
            d_bias = numpy.sum(node.d_bias, axis=0)
            if not hasattr(node, 'prev_d_bias'):
                node.prev_d_bias = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_bias'):
                node.delta_bias = numpy.ones(node.bias.shape) * self.delta_init

            if not hasattr(node, 'prev_bias_update'):
                node.prev_bias_update = numpy.zeros(node.bias.shape)

            self.upd(node.bias, d_bias, node.prev_d_bias, node.delta_bias,
                     node.prev_bias_update, node.get_arity())

        if node.tune_weights and hasattr(node, 'd_weights'):
            upd = True
            d_weights = numpy.sum(node.d_weights, axis=0)
            if not hasattr(node, 'prev_d_weights'):
                node.prev_d_weights = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_weights'):
                node.delta_weights = numpy.ones(node.weights.shape) *\
                                     self.delta_init

            if not hasattr(node, 'prev_weight_update'):
                node.prev_weight_update = numpy.zeros(node.weights.shape)

            self.upd(node.weights, d_weights, node.prev_d_weights,
                     node.delta_weights, node.prev_weight_update,
                     node.get_arity())

        if upd:
            self.updated = True
            node.notify_change()

    def update(self, root: WeightedNode, error, prev_error):
        self.error_worsened = error > prev_error
        super().update(root, error, prev_error)


class IRpropMinus(RpropBase):
    """This is the improved Rprop algorithm without weight-backtracking, also
    called iRprop\ :sup:`-` [Igel2000].

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """

    def upd(self, w, d, prev_d, delta, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
                prev_d[i] = 0
            w[i] -= s[i] * delta[i]
            prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and hasattr(node, 'd_bias'):
            upd = True
            d_bias = numpy.sum(node.d_bias, axis=0)
            if not hasattr(node, 'prev_d_bias'):
                node.prev_d_bias = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_bias'):
                node.delta_bias = numpy.ones(node.bias.shape) * self.delta_init

            self.upd(node.bias, d_bias, node.prev_d_bias, node.delta_bias,
                     node.get_arity())

        if node.tune_weights and hasattr(node, 'd_weights'):
            upd = True
            d_weights = numpy.sum(node.d_weights, axis=0)
            if not hasattr(node, 'prev_d_weights'):
                node.prev_d_weights = numpy.zeros(node.get_arity())

            if not hasattr(node, 'delta_weights'):
                node.delta_weights = numpy.ones(node.weights.shape) *\
                                     self.delta_init

            self.upd(node.weights, d_weights, node.prev_d_weights,
                     node.delta_weights, node.get_arity())

        if upd:
            self.updated = True
            node.notify_change()
