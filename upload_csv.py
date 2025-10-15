#!/usr/bin/env python3
"""Upload CSV to S3"""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

bucket = os.getenv('S3_BUCKET', 'nba-foul-dataset-oh')
csv_file = 'data/metadata/nba_fouls_multi-season_1213clips_20251015_094649.csv'

s3.upload_file(csv_file, bucket, 'metadata/dataset.csv')
print(f'✓ Uploaded to s3://{bucket}/metadata/dataset.csv')
