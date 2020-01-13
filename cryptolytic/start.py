import json
import logging
import sys
from dotenv import load_dotenv

def start_logging():
    logging.basicConfig(filename="log.txt")

    # also print to stderr
    stderrLogger=logging.StreamHandler()
    stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderrLogger)

    w = 20
    logging.debug(' '*w)
    logging.debug('-'*w) 
    logging.debug('LOG START')
    logging.debug('-'*w) 

def init():
    load_dotenv(verbose=True)
    start_logging()
