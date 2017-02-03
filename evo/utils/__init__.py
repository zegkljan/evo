# -*- coding: utf-8 -*-
"""This package contains support modules for evolutionary algorithms.
"""

import collections
import logging

import numpy

LOG = logging.getLogger(__name__)


def _partition(vector, left, right, pivot_index, cmp):
    pivot_value = vector[pivot_index]
    vector[pivot_index], vector[right] = vector[right], vector[pivot_index]
    store_index = left
    for i in range(left, right):
        if cmp(vector[i], pivot_value):
            vector[store_index], vector[i] = vector[i], vector[store_index]
            store_index += 1
    vector[right], vector[store_index] = vector[store_index], vector[right]
    return store_index


def select(vector, k, left=None, right=None, cmp=None):
    """Returns the k-th smallest element of vector within vector[left:right + 1]
    inclusive.

    :param vector: the vector of elements to be searched
    :param int k: the rank of desired element
    :param int left: start index of the searched sub-sequence of the vector; if
        ``None`` value of ``0`` is used
    :param int right: end index (inclusive) of the searched sub-sequence of the
        vector; if ``None`` value of ``len(vector) - 1`` is used
    :param cmp: a callable taking two arguments (from the vector) returning
        ``True`` if and only if the first argument is considered to precede the
        second argument; if ``None`` the ``<`` operator will be used
    """
    if left is None:
        left = 0
    if right is None:
        right = len(vector) - 1
    if k > right - left:
        raise ValueError("The rank is greater than the number of searched "
                         "elements (k={}, right={}, left={}).".format(k,
                                                                      right,
                                                                      left))
    if cmp is None:
        def cmp(a, b):
            return a < b
    while True:
        pivot_index = (right - left) // 2 + left
        pivot_new_index = _partition(vector, left, right, pivot_index, cmp)
        pivot_dist = pivot_new_index - left
        if pivot_dist == k:
            return vector[pivot_new_index]
        elif k < pivot_dist:
            right = pivot_new_index - 1
        else:
            k -= pivot_dist + 1
            left = pivot_new_index + 1


def nested_update(base: dict, update: dict, inplace=True) -> dict:
    if not inplace:
        base = dict(base)
    for k, v in update.items():
        if k not in base:
            base[k] = v
        elif isinstance(base[k], dict) and isinstance(v, dict):
            nested_update(base[k], v)
        else:
            base[k] = v
    return base


def efficient_column_stack(*args, min_rows: int=1):
    l = min_rows
    for a in args:
        try:
            if a.size < min_rows:
                raise ValueError('Input arrays must not be smaller than '
                                 'minimum number of rows required.')
            l = max(l, a.size)
        except AttributeError:
            pass
    ars = []
    for a in args:
        try:
            s = a.size
        except AttributeError:
            s = 1
        if s == 1:
            ars.append(numpy.repeat(a, l)[numpy.newaxis, :])
        else:
            ars.append(numpy.array(a, copy=False, subok=True, ndmin=2))
    result = numpy.array(numpy.vstack(ars).T, copy=True)
    del ars
    del args
    return result


def broadcast_column_stack(*args, min_rows: int=1):
    return numpy.column_stack(numpy.broadcast(*args)).T

column_stack = efficient_column_stack


def flatten(l):
    for e in l:
        if (isinstance(e, collections.Iterable) and
                not isinstance(e, (str, bytes, numpy.ndarray))):
            yield from flatten(e)
        else:
            yield e
