import signal

def timeout_handler(signum, frame):
    raise TimeoutError()

def timeout(fn, time, handler=timeout_handler):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time)
