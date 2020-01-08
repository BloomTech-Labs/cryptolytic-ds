import json
import os
from dotenv import load_dotenv

def init():
    # using test environment
    load_dotenv(verbose=True, dotenv_path='tests/test.env')
    print(os.environ['POSTGRES_DBNAME'])
