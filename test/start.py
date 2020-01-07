import json
from dotenv import load_dotenv

def init():
    # using test environment
    load_dotenv(verbose=True, dotenv_path='test/test.env')
