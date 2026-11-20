"""Download a prefix from the backup bucket.  python scripts/fetch_s3.py BUCKET PREFIX OUT_DIR"""
import os, sys, boto3
from boto3.session import Config
s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], region_name="auto",
                  config=Config(s3={"addressing_style": "path"}))
bucket, prefix, out = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3]
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for o in page.get("Contents", []):
        dst = os.path.join(out, o["Key"][len(prefix):]); os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        s3.download_file(bucket, o["Key"], dst); n += 1
print("downloaded", n, "files to", out)
