"""Upload files/directories to the backup bucket.

  python scripts/backup_s3.py BUCKET PREFIX PATH[=NAME]...

Each PATH lands under PREFIX/<its last path component>/ - give PATH=NAME to choose the name
(needed when several directories share a last component such as "final").
"""
import os, sys, boto3
from boto3.session import Config
s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], region_name="auto",
                  config=Config(s3={"addressing_style": "path"}))
bucket, prefix = sys.argv[1], sys.argv[2].strip("/")
n = b = 0
seen = set()
for spec in sys.argv[3:]:
    path, _, name = spec.partition("=")
    path = path.rstrip("/")
    name = name or os.path.basename(path)
    if name in seen:
        sys.exit(f"two paths would both be stored as {prefix}/{name} - give PATH=NAME")
    seen.add(name)
    files = [path] if os.path.isfile(path) else [os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs]
    for f in files:
        key = f"{prefix}/{name}" if os.path.isfile(path) else f"{prefix}/{name}/{os.path.relpath(f, path)}"
        s3.upload_file(f, bucket, key); n += 1; b += os.path.getsize(f)
print(f"uploaded {n} files, {b/1e9:.2f} GB to s3://{bucket}/{prefix}/")
