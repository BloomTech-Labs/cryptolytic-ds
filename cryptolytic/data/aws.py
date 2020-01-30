from cryptolytic import session

bucket_name = 'crypto-buckit'


def download_file(path):
    """
    Get file from s3
    """
    s3 = session.resource('s3')
    s3_object = s3.Object(bucket_name=bucket_name, key=path)
    return s3_object.download_file(path)


def upload_file(path):
    """
    Put file on s3
    """
    s3 = session.client('s3')
    return s3.upload_file(path, bucket_name, path)

