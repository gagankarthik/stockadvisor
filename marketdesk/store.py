"""Artifact / state persistence over local filesystem or S3.

Lambda's own filesystem is read-only except for `/tmp`, so anything that must
survive between invocations (the trained model, the daily snapshot, saved
plans) lives in an `ArtifactStore`. One small abstraction with two backends:

- ``file://<path>`` or a bare path  -> local filesystem (dev, /tmp).
- ``s3://<bucket>/<prefix>``         -> S3 (production / Lambda).

boto3 is imported lazily so local development never needs AWS installed.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class ArtifactStore:
    """Key/blob storage. Keys are relative paths like ``model.joblib``."""

    def get_bytes(self, key: str) -> bytes | None:
        raise NotImplementedError

    def put_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def last_modified(self, key: str) -> float | None:
        """Unix timestamp of the object, or None if it does not exist."""
        raise NotImplementedError

    # ---- JSON helpers shared by both backends ----
    def get_json(self, key: str) -> Any | None:
        raw = self.get_bytes(key)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def put_json(self, key: str, obj: Any) -> None:
        self.put_bytes(key, json.dumps(obj, indent=2, default=str).encode("utf-8"))


@dataclass
class LocalStore(ArtifactStore):
    root: str

    def __post_init__(self) -> None:
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def last_modified(self, key: str) -> float | None:
        path = self._path(key)
        return os.path.getmtime(path) if os.path.exists(path) else None


@dataclass
class S3Store(ArtifactStore):
    bucket: str
    prefix: str = ""

    def __post_init__(self) -> None:
        import boto3  # lazy: only needed in the S3 path

        self._s3 = boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix.rstrip('/')}/{key}" if self.prefix else key

    def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._key(key))
            return obj["Body"].read()
        except ClientError:
            return None

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._key(key),
                            Body=io.BytesIO(data).getvalue())

    def exists(self, key: str) -> bool:
        return self.last_modified(key) is not None

    def last_modified(self, key: str) -> float | None:
        from botocore.exceptions import ClientError

        try:
            head = self._s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return head["LastModified"].timestamp()
        except ClientError:
            return None


def build_store(uri: str) -> ArtifactStore:
    """Construct the right backend from an artifact URI."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return S3Store(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    if parsed.scheme in ("file", ""):
        # file://relative, file:///abs, or a bare path
        path = (parsed.netloc + parsed.path) if parsed.scheme == "file" else uri
        return LocalStore(root=path or ".")
    raise ValueError(f"Unsupported artifact URI scheme: {uri!r}")
