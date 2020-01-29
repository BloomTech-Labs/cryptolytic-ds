import boto3
import pandas as pd
from io import StringIO

# TODO connect to aws with credentials from env file, should 
# be called from start.init()
def connect_to_s3():
    s3 = boto3.client('s3')
    return s3
