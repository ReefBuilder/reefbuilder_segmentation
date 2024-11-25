import os
from functools import wraps


def expand_paths(*path_args):
    """
    A decorator to validate and expand specified path arguments.

    Args:
        path_args: Names of the arguments that should be validated as paths.

    Returns:
        A decorated function that ensures specified arguments are valid paths.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args to a mutable dictionary
            from inspect import signature

            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Process the specified path arguments
            for path_arg in path_args:
                if path_arg in bound_args.arguments:
                    path_value = bound_args.arguments[path_arg]
                    if path_value is not None:
                        # Apply expanduser and abspath
                        path_value = os.path.abspath(os.path.expanduser(path_value))
                        bound_args.arguments[path_arg] = path_value

            # Call the original function with modified arguments
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
