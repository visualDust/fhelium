"""Inline Python callable objects through descriptor-aware captured state reads."""

from __future__ import annotations

import inspect
import types
from collections.abc import ItemsView, KeysView, ValuesView
from typing import Any, Callable, cast

from .._errors import CaptureError


def python_callable(source) -> Callable[..., Any]:
    """Return a synchronous Python function or bound method for one callable."""
    method = (
        source
        if isinstance(source, (types.FunctionType, types.MethodType))
        else getattr(source, "__call__", None)
    )
    function = (
        method.__func__ if isinstance(method, types.MethodType) else method
    )
    if (
        not isinstance(function, types.FunctionType)
        or inspect.iscoroutinefunction(function)
        or inspect.isasyncgenfunction(function)
    ):
        raise CaptureError("Callable capture requires synchronous Python code")
    return cast(Callable[..., Any], method)


def _binding(proxy):
    return object.__getattribute__(proxy, '_StaticObject__binding')


def unwrap(value) -> Any:
    """Recover an object's identity for capture-time Python metadata operations."""
    return _binding(value)[0] if isinstance(value, StaticObject) else value


def _method(proxy, name, *args, **kwargs):
    return _attribute(proxy, name)(*args, **kwargs)


def _attribute(proxy, name):
    original, capture = _binding(proxy)
    if name == '__class__':
        return type(original)
    if type(original) in (dict, types.MappingProxyType):
        if name == 'get':
            return lambda key, default=None: (
                proxy[key] if key in original else default
            )
        if name == 'keys':
            return lambda: KeysView(proxy)
        if name == 'values':
            return lambda: ValuesView(proxy)
        if name == 'items':
            return lambda: ItemsView(proxy)
        if name == 'copy':
            return lambda: {key: proxy[key] for key in original}
        if name in {
            'update',
            'clear',
            'pop',
            'popitem',
            'setdefault',
            '__setitem__',
            '__delitem__',
        }:
            raise CaptureError(
                'Captured computation cannot mutate bound mapping state'
            )
    try:
        descriptor = inspect.getattr_static(original, name)
    except AttributeError:
        dynamic = inspect.getattr_static(type(original), '__getattr__', None)
        if isinstance(dynamic, types.FunctionType):
            return capture.function(dynamic)(proxy, name)
        raise
    if isinstance(descriptor, property):
        if descriptor.fget is None:
            raise AttributeError(name)
        return capture.function(descriptor.fget)(proxy)
    if isinstance(descriptor, staticmethod):
        return capture.substitute(descriptor.__func__)
    if isinstance(descriptor, classmethod):
        owner = original if isinstance(original, type) else type(original)
        return types.MethodType(
            capture.function(descriptor.__func__), capture.object(owner)
        )
    # Ordinary instance fields can themselves contain a Python function; they
    # do not acquire descriptor binding merely because their value is callable.
    dictionary = inspect.getattr_static(type(original), '__dict__', None)
    instance_fields = (
        dictionary.__get__(original, type(original))
        if isinstance(dictionary, types.GetSetDescriptorType)
        and not isinstance(original, type)
        else {}
    )
    if name in instance_fields and descriptor is instance_fields[name]:
        return capture.substitute(descriptor)
    if isinstance(descriptor, types.FunctionType):
        function = capture.function(descriptor)
        return (
            function
            if isinstance(original, type)
            else types.MethodType(function, proxy)
        )
    if isinstance(descriptor, types.MemberDescriptorType) or (
        name == '__dict__'
        and isinstance(descriptor, types.GetSetDescriptorType)
    ):
        return capture.substitute(descriptor.__get__(original, type(original)))
    if hasattr(type(descriptor), '__get__'):
        raise CaptureError(
            f'Callable capture does not support descriptor {type(descriptor).__name__} for {name!r}'
        )
    return capture.substitute(descriptor)


class StaticObject:
    """Interpret supported Python descriptors on a read-only captured receiver.

    Internal binding storage is accessed through object primitives. Every normal
    attribute lookup resolves against the user's object, including names also
    used internally by the frontend. Tensor fields become materials when read.
    """

    __slots__ = ('__binding',)

    def __init__(self, value: object, capture: Any) -> None:
        object.__setattr__(self, '_StaticObject__binding', (value, capture))

    def __getattribute__(self, name: str) -> Any:
        return _attribute(self, name)

    def __setattr__(self, name, value):
        raise CaptureError(
            'Captured computation cannot mutate bound object state'
        )

    def __call__(self, *args, **kwargs):
        return _method(self, '__call__', *args, **kwargs)

    def __hash__(self):
        return hash(unwrap(self))

    def __eq__(self, other):
        original, capture = _binding(self)
        method = type(original).__eq__
        if isinstance(method, types.FunctionType):
            return capture.function(method)(self, other)
        return original == unwrap(other)


class StaticMapping(StaticObject):
    """Read mapping entries lazily and preserve the original mapping's methods."""

    def __getitem__(self, key):
        original, capture = _binding(self)
        if type(original) in (dict, types.MappingProxyType):
            return capture.substitute(original[key])
        return _method(self, '__getitem__', key)

    def __iter__(self):
        original, _ = _binding(self)
        return (
            iter(original)
            if type(original) in (dict, types.MappingProxyType)
            else _method(self, '__iter__')
        )

    def __len__(self):
        original, _ = _binding(self)
        return (
            len(original)
            if type(original) in (dict, types.MappingProxyType)
            else _method(self, '__len__')
        )

    def __contains__(self, key):
        original, _ = _binding(self)
        return (
            key in original
            if type(original) in (dict, types.MappingProxyType)
            else _method(self, '__contains__', key)
        )

    def __setitem__(self, key, value):
        raise CaptureError(
            'Captured computation cannot mutate bound mapping state'
        )

    def __delitem__(self, key):
        raise CaptureError(
            'Captured computation cannot mutate bound mapping state'
        )
