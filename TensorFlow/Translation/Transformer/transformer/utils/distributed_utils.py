import horovod.tensorflow as hvd

def suppress_output():
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print_master(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
        builtin_print(*args, **kwargs)

    def print(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
            if force:
                builtin_print(*args, **kwargs)
    if(hvd.rank()==0):
        __builtin__.print = print_master
    else:
        __builtin__.print = print
