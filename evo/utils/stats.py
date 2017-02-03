# -*- coding: utf-8 -*-
"""This module provides means to report data from the EA runs other than
provided by standard logging.
"""

import csv


class Stats(object):
    """A base class for saving statistics of a run of an evolutionary
    algorithm.

    This class itself does nothing, you need to derive from it and implement
    the desired behavior.
    """

    def report_data(self, data):
        """Saves arbitrary data.

        :param data: data to be reported
        """
        pass


class CsvStats(Stats):
    """An implementation of :class:`.Stats` that treats the data as lists and
    stores the elements as rows of a CSV.
    """

    def __init__(self, file, delimiter=','):
        """
        :param file: a name of a file or a file-like object used to save the CSV
            into
        """
        if isinstance(file, str):
            self._file = open(file, 'w', newline='')
        else:
            self._file = file
        self.file = csv.writer(self._file, delimiter=delimiter,
                               quoting=csv.QUOTE_ALL)

    def report_data(self, data):
        self.file.writerow(data)
