hvd_global_object = None

def init(use_horovod: bool = False):
    global hvd_global_object
    if use_horovod:
        import horovod.tensorflow as hvd
        hvd.init()
        hvd_global_object = hvd
    else:
        class _DummyWrapper:
            def rank(self): return 0
            def size(self): return 1
            def local_rank(self): return 0
            def local_size(self): return 1
        hvd_global_object = _DummyWrapper()


def size():
    global hvd_global_object
    return hvd_global_object.size()

def rank():
    global hvd_global_object
    return hvd_global_object.rank()

def local_rank():
    global hvd_global_object
    return hvd_global_object.local_rank()

def local_size():
    global hvd_global_object
    return hvd_global_object.local_size()