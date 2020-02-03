from datetime import datetime
import time


def convert_datetime(t):
    """Convert string to unix time stamp if not. Currently handles %d-%m-%Y
       Consider using something more convenient like the arrow library."""
    try:
        if isinstance(t, str):
            converted = datetime.strptime(t, '%d-%m-%Y')
            # to get time in seconds:
            t = int(time.mktime(converted.timetuple()))
        return t
    except Exception as e:
        print(e)
        return None
