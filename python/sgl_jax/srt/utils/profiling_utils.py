import functools

import jax


def named_scope(arg=None, *, name: str | None = None):
    """Decorator to add a JAX named_scope based on the first argument.

    Usage:
        @named_scope
        def fn(...): ...

        @named_scope(\"explicit_name\")
        def fn(...): ...

        @named_scope(name=\"explicit_name\")
        def fn(...): ...
    """

    # Allow positional name: @named_scope("foo")
    if isinstance(arg, str) and name is None:
        name = arg
        arg = None

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if name:
                scope = name
            elif args:
                obj = args[0]
                if hasattr(obj, "name"):
                    scope = obj.name
                elif getattr(type(obj), "__module__", "") != "builtins":
                    scope = type(obj).__name__
                else:
                    scope = fn.__qualname__
            else:
                scope = fn.__qualname__

            with jax.named_scope(scope):
                return fn(*args, **kwargs)

        return wrapper

    if callable(arg):
        # Called as @named_scope without arguments.
        return decorator(arg)
    else:
        # Called as @named_scope(...) with optional name.
        return decorator
