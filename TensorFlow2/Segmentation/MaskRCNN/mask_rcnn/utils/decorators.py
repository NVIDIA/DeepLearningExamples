#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import atexit
import functools
import inspect
import signal
import wrapt

__all__ = ["atexit_hook"]

_executed_exit_fns = set()
_registered_exit_fns = set()
_registered_objects = set()


def register_atexit_fn(fun=None, signals=None, logfun=lambda s: print(s, file=sys.stderr)):
    """Register a function which will be executed on "normal"
    interpreter exit or in case one of the `signals` is received
    by this process (differently from atexit.register()).
    Also, it makes sure to execute any other function which was
    previously registered via signal.signal(). If any, it will be
    executed after our own `fun`.

    Functions which were already registered or executed via this
    function will be ignored.

    Note: there's no way to escape SIGKILL, SIGSTOP or os._exit(0)
    so don't bother trying.

    You can use this either as a function or as a decorator:

        @register_atexit_fn
        def cleanup():
            pass

        # ...or

        register_atexit_fn(cleanup)

    Note about Windows: I tested this some time ago and didn't work
    exactly the same as on UNIX, then I didn't care about it
    anymore and didn't test since then so may not work on Windows.

    Parameters:

    - fun: a callable
    - signals: a list of signals for which this function will be
      executed (default SIGTERM)
    - logfun: a logging function which is called when a signal is
      received. Default: print to standard error. May be set to
      None if no logging is desired.
    """
    '''
    Source: https://github.com/torvalds/linux/blob/master/include/linux/signal.h

    *    +--------------------+-----------------+
    *    |  POSIX signal      |  default action |
    *    +--------------------+-----------------+
    *    |  SIGHUP            |    terminate    |
    *    |  SIGINT            |    terminate    |
    *    |  SIGQUIT           |    coredump     |
    *    |  SIGILL            |    coredump     |
    *    |  SIGTRAP           |    coredump     |
    *    |  SIGABRT/SIGIOT    |    coredump     |
    *    |  SIGBUS            |    coredump     |
    *    |  SIGFPE            |    coredump     |
    *    |  SIGKILL           |    terminate(+) |
    *    |  SIGUSR1           |    terminate    |
    *    |  SIGSEGV           |    coredump     |
    *    |  SIGUSR2           |    terminate    |
    *    |  SIGPIPE           |    terminate    |
    *    |  SIGALRM           |    terminate    |
    *    |  SIGTERM           |    terminate    |
    *    |  SIGCHLD           |    ignore       |
    *    |  SIGCONT           |    ignore(*)    |
    *    |  SIGSTOP           |    stop(*)(+)   |
    *    |  SIGTSTP           |    stop(*)      |
    *    |  SIGTTIN           |    stop(*)      |
    *    |  SIGTTOU           |    stop(*)      |
    *    |  SIGURG            |    ignore       |
    *    |  SIGXCPU           |    coredump     |
    *    |  SIGXFSZ           |    coredump     |
    *    |  SIGVTALRM         |    terminate    |
    *    |  SIGPROF           |    terminate    |
    *    |  SIGPOLL/SIGIO     |    terminate    |
    *    |  SIGSYS/SIGUNUSED  |    coredump     |
    *    |  SIGSTKFLT         |    terminate    |
    *    |  SIGWINCH          |    ignore       |
    *    |  SIGPWR            |    terminate    |
    *    |  SIGRTMIN-SIGRTMAX |    terminate    |
    *    +--------------------+-----------------+
    *    |  non-POSIX signal  |  default action |
    *    +--------------------+-----------------+
    *    |  SIGEMT            |    coredump     |
    *    +--------------------+-----------------+
    '''

    if signals is None:
        signals = [signal.SIGTERM]

    def stringify_sig(signum):
        if sys.version_info < (3, 5):
            smap = dict([(getattr(signal, x), x) for x in dir(signal) if x.startswith('SIG')])
            return smap.get(signum, signum)
        else:
            return signum

    def fun_wrapper():
        if fun not in _executed_exit_fns:
            try:
                fun()
            finally:
                _executed_exit_fns.add(fun)

    def signal_wrapper(signum=None, frame=None):
        if signum is not None:
            if logfun is not None:
                logfun("signal {} received by process with PID {}".format(stringify_sig(signum), os.getpid()))
        fun_wrapper()
        # Only return the original signal this process was hit with
        # in case fun returns with no errors, otherwise process will
        # return with sig 1.
        if signum is not None:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            # XXX - should we do the same for SIGTERM / SystemExit?
            sys.exit(signum)

    def register_fun(fun, signals):
        if not callable(fun):
            raise TypeError("{!r} is not callable".format(fun))
        set([fun])  # raise exc if obj is not hash-able

        signals = set(signals)
        for sig in signals:
            # Register function for this signal and pop() the previously
            # registered one (if any). This can either be a callable,
            # SIG_IGN (ignore signal) or SIG_DFL (perform default action
            # for signal).
            old_handler = signal.signal(sig, signal_wrapper)
            if old_handler not in (signal.SIG_DFL, signal.SIG_IGN):
                # ...just for extra safety.
                if not callable(old_handler):
                    continue
                # This is needed otherwise we'll get a KeyboardInterrupt
                # strace on interpreter exit, even if the process exited
                # with sig 0.
                if (sig == signal.SIGINT and old_handler is signal.default_int_handler):
                    continue
                # There was a function which was already registered for this
                # signal. Register it again so it will get executed (after our
                # new fun).
                if old_handler not in _registered_exit_fns:
                    atexit.register(old_handler)
                    _registered_exit_fns.add(old_handler)

        # This further registration will be executed in case of clean
        # interpreter exit (no signals received).
        if fun not in _registered_exit_fns or not signals:
            atexit.register(fun_wrapper)
            _registered_exit_fns.add(fun)

    # This piece of machinery handles 3 usage cases. register_atexit_fn()
    # used as:
    # - a function
    # - a decorator without parentheses
    # - a decorator with parentheses
    if fun is None:

        @functools.wraps
        def outer(fun):
            return register_fun(fun, signals)

        return outer
    else:
        register_fun(fun, signals)
        return fun


def atexit_hook(*args, **kwargs):

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):

        if not hasattr(wrapped, "__atexit__"):
            raise AttributeError("The class `%s` does not have an `__atexit__` method" % wrapped.__name__)

        def _func():
            if instance is None:
                if inspect.isclass(wrapped):
                    # Decorator was applied to a class.
                    return wrapped(*args, **kwargs)
                else:
                    # Decorator was applied to a function or staticmethod.
                    return wrapped(*args, **kwargs)
            else:
                if inspect.isclass(instance):
                    # Decorator was applied to a classmethod.
                    return wrapped(*args, **kwargs)
                else:
                    # Decorator was applied to an instancemethod.
                    return wrapped(*args, **kwargs)

        _impl = _func()

        object_id = hex(id(_impl))

        if object_id not in _registered_objects:
            register_atexit_fn(fun=_impl.__atexit__, signals=[signal.SIGTERM, signal.SIGINT])
            _registered_objects.add(object_id)

        return _impl

    return wrapper(*args, **kwargs)
