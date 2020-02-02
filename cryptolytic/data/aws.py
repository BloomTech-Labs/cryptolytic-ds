from cryptolytic import session
from cryptolytic.start import init
import os

bucket_name = 'crypto-buckit'


def download_file(path):
    """
    Get file from s3
    """
    init()
    s3 = session.resource('s3')
    s3_object = s3.Object(bucket_name=bucket_name, key=path)
    return s3_object.download_file(path)


def upload_file(path):
    """
    Put file on s3
    """
    init()
    s3 = session.client('s3')
    return s3.upload_file(path, bucket_name, path)


def get_path(folder_name, model_type, exchange_id, trading_pair, ext):
    """
    Example models/model_trade_binance_eth_usd.pkl
            preds/model_trade_binance_eth_usd.csv
    """
    aws_folder = os.path.join('aws', folder_name)
    if not os.path.exists(aws_folder):
        os.mkdir(aws_folder)
    return os.path.join(aws_folder, f'model_{model_type}_{exchange_id}_{trading_pair}{ext}').replace('\\', '/')
