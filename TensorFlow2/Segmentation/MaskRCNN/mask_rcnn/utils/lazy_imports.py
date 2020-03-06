#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2006-2011, NIPY Developers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NIPY Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Source: https://github.com/nipy/nitime/blob/c8eb314/nitime/lazyimports.py

"""This module provides lazy import functionality to improve the import
performance of nitime. For example, some parts of nitime leverage and import
matplotlib, which is quite a big package, yet most of the nitime code does not
depend on matplotlib. By lazily-loading a module, we defer the overhead of
importing it until the first time it is actually used, thereby speeding up
nitime imports.

A generic :class:`LazyImport` class is implemented which takes the module name
as a parameter, and acts as a proxy for that module, importing it only when
the module is used, but effectively acting as the module in every other way
(including inside IPython with respect to introspection and tab completion)
with the *exception* of reload() - reloading a :class:`LazyImport` raises an
:class:`ImportError`.

Commonly used nitime lazy imports are also defined in :mod:`nitime.lazy`, so
they can be reused throughout nitime.
"""

import os
import sys
import types


class LazyImport(types.ModuleType):
    """
    This class takes the module name as a parameter, and acts as a proxy for
    that module, importing it only when the module is used, but effectively
    acting as the module in every other way (including inside IPython with
    respect to introspection and tab completion) with the *exception* of
    reload()- reloading a :class:`LazyImport` raises an :class:`ImportError`.

    >>> mlab = LazyImport('matplotlib.mlab')

    No import happens on the above line, until we do something like call an
    ``mlab`` method or try to do tab completion or introspection on ``mlab``
    in IPython.

    >>> mlab
    <module 'matplotlib.mlab' will be lazily loaded>

    Now the :class:`LazyImport` will do an actual import, and call the dist
    function of the imported module.

    >>> mlab.dist(1969,2011)
    42.0
    """

    def __getattribute__(self, x):
        # This method will be called only once, since we'll change
        # self.__class__ to LoadedLazyImport, and __getattribute__ will point
        # to module.__getattribute__

        name = object.__getattribute__(self, '__name__')
        __import__(name)

        # if name above is 'package.foo.bar', package is returned, the docs
        # recommend that in order to get back the full thing, that we import
        # and then lookup the full name is sys.modules, see:
        # http://docs.python.org/library/functions.html#__import__

        module = sys.modules[name]

        # Now that we've done the import, cutout the middleman and make self
        # act as the imported module

        class LoadedLazyImport(types.ModuleType):
            __getattribute__ = module.__getattribute__
            __repr__ = module.__repr__

        object.__setattr__(self, '__class__', LoadedLazyImport)

        # The next line will make "reload(l)" a silent no-op
        return module.__getattribute__(x)

    def __repr__(self):
        return "<module '%s' will be lazily loaded>" % object.__getattribute__(self, '__name__')


if 'READTHEDOCS' in os.environ:
    lazy_doc = """
               WARNING: To get Sphinx documentation to build we disable
               LazyImports, which makes Sphinx incorrectly report this
               class as having a base class of object. In reality,
               :class:`LazyImport`'s base class is
               :class:`types.ModuleType`.
               """

    lazy_doc += LazyImport.__doc__

    class LazyImport(object):
        __doc__ = lazy_doc

        def __init__(self, x):
            __import__(x)
            self.module = sys.modules[x]

        def __getattr__(self, x):
            return self.module.__getattribute__(x)
