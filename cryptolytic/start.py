import json
import logging
from dotenv import load_dotenv

def start_logging():
    logging.basicConfig(level=logging.DEBUG, filename="log.txt")
    w = 20
    logging.debug(' '*w)
    logging.debug('-'*w) 
    logging.debug('LOG START')
    logging.debug('-'*w) 

def init():
    load_dotenv(verbose=True)
    start_logging()
