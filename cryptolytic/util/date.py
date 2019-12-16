from datetime import datetime
import time

def convert_datetime(t):
    """Convert value to unix time stamp if not. Currently handles %d-%m-%Y"""
    try:
        result = t
        if type(t)==type(""):
            converted = datetime.strptime(t, '%d-%m-%Y')
            # to get time in seconds:
            t = int(time.mktime(converted.timetuple()))
        return t
    except Exception as e:
        print(e)
        return None