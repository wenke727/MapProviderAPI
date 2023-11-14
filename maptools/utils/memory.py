
def create_memory_cache(cachedir='./joblib_cache', verbose=0):
    """
    Creates a memory cache object for caching function results with joblib.

    This function is intended to be used for creating a cache directory where joblib
    can store the results of expensive function calls, so that subsequent calls with
    the same arguments can return cached results quickly without re-computation.

    Parameters:
    - cachedir (str): The directory where the cache should be stored. If the directory
      does not exist, it will be created. Defaults to './joblib_cache'.
    - verbose (int): The verbosity level of joblib messages. If set to a number greater
      than 0, joblib will print messages related to caching. Defaults to 0 (no output).

    Returns:
    Memory: A configured Memory object that can be used to decorate functions for caching.

    Usage:
    To use this cache, decorate a function with the `@memory.cache` decorator, where
    `memory` is the object returned by this function. For example:

    >>> memory = create_memory_cache(cachedir='my_cache', verbose=1)
    >>> @memory.cache
    ... def my_expensive_function(arg1, arg2):
    ...     # Expensive computation here
    ...     return result

    Now, calling `my_expensive_function(arg1, arg2)` will compute and cache the result.
    Subsequent calls with the same arguments will return the cached result instantly.

    Notes:
    The function you wish to cache must take only hashable arguments because joblib
    relies on hashing to lookup the cache. This means that you cannot use arguments
    like lists or dictionaries unless they are wrapped inside a hashable type like
    a tuple or a frozenset.
    """
    from joblib import Memory
    memory = Memory(cachedir, verbose=verbose)

    return memory


MEMORY = create_memory_cache('../cache', verbose=1)
