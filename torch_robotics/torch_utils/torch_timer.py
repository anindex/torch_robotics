# Adapted from https://github.com/brouberol/contexttimer
import time
from time import perf_counter

import torch


class Timer(object):
    """ A timer as a context manager

    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.

    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Keyword arguments:
        output -- if True, print output after exiting context.
                  if callable, pass output to callable.
        format -- str.format string to be used for output; default "took {} seconds"
        prefix -- string to prepend (plus a space) to output
                  For convenience, if you only specify this, output defaults to True.
    """

    def __init__(self, timer=perf_counter, factor=1,
                 output=None, fmt="took {:.3f} seconds", prefix="",
                 sync_cuda=True
                 ):
        self.timer = timer
        self.factor = factor
        self.output = output
        self.fmt = fmt
        self.prefix = prefix
        self.end = None
        self.sync_cuda = True if sync_cuda and torch.cuda.is_available() else False

    def __call__(self):
        """ Return the current time """
        if self.sync_cuda:
            torch.cuda.synchronize()
        return self.timer()

    def __enter__(self):
        """ Set the start time """
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

        if self.prefix and self.output is None:
            self.output = True

        if self.output:
            output = " ".join([self.prefix, self.fmt.format(self.elapsed)])
            if callable(self.output):
                self.output(output)
            else:
                print(output)

    def __str__(self):
        return f'{self.elapsed:.3f}'

    @property
    def elapsed(self):
        """ Return the current elapsed time since start

        If the `elapsed` property is called in the context manager scope,
        the elapsed time between start and property access is returned.
        However, if it is accessed outside of the context manager scope,
        it returns the elapsed time between entering and exiting the scope.

        The `elapsed` property can thus be accessed at different points within
        the context manager scope, to time different parts of the block.

        """
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor

