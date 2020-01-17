import signal
import multiprocessing 

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
