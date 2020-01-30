import json
import logging
import sys
from dotenv import load_dotenv
import cryptolytic
import os
import boto3


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
    cryptolytic.session = boto3.session.Session(aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
                                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
