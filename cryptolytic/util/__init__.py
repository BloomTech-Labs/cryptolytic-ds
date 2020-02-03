import functools
import signal
import multiprocessing
import pprint

pprint = pprint.PrettyPrinter().pprint


def timeout_handler(signum, frame):
    raise TimeoutError()


def timeout(fn, time, timeout_handler=None, success_handler=None):
    """fn: function to call
       time: time to wait before cancelling the thread it was put on
       timeout_handler: function to call if thread was cancelled
       success_handler: function to call if thread was not cancelled"""
    try:
        p = multiprocessing.Process(target=fn)
        p.start()  # start process on fn
        p.join(time)  # wait x seconds
        # if thread is still active
        if p.is_alive():
            # terminate
            p.terminate()
            p.join()
            if timeout_handler is not None:
                timeout_handler()
        else:
            if success_handler is not None:
                success_handler()
    except Exception as e:
        print(e)
        return


def bdir(x):
    "better dir command, filters out things beginning with _"
    return list(filter(lambda x: not x.startswith('_'), dir(x)))


def compose(*functions):
    """Simple function composition"""
    def compose2(f, g):
        return lambda *a, **k: f(g(*a, **k))
    return functools.reduce(compose2, functions)


def mapl(f, coll):
    return list(map(f, coll))


def filterl(f, coll):
    return list(filter(f, coll))


def first(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return x[0]


def dict_matches(cond, b):
    return set(cond.items()).issubset(set(b.items()))


def select_keys(d, keys):
    return {k: d[k] for k in keys if k in d}


class adict(dict):
    def __init__(self, *args, **kwargs):
        super(adict, self).__init__(*args, **kwargs)
        self.__dict__ = self
