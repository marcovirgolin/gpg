from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps


def threading_timeoutable(default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout = kwargs.pop("timeout", None)
            if timeout is None:
                return func(*args, **kwargs)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except FutureTimeoutError:
                    return default

        return wrapper

    return decorator
