"""Tests for the S3 / S3-compatible storage provider boundary."""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import pytest
from botocore.exceptions import ClientError

from fivccliche.services.implements.storages_s3 import S3StorageProvider
from fivccliche.services.interfaces.storages import StorageError


class FakeConfigSession:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get_value(self, key: str) -> str | None:
        return self.values.get(key)


class FakeConfig:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get_session(self, name: str) -> FakeConfigSession:
        return FakeConfigSession(self.values if name == "S3" else {})


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.delete_errors: set[str] = set()

    def generate_presigned_url(self, *args: Any, **kwargs: Any) -> str:
        # Mirror boto3: positional ClientMethod, keyword Params / ExpiresIn.
        method = args[0] if args else kwargs.get("ClientMethod", "")
        params = kwargs.get("Params") or {}
        expires_in = kwargs.get("ExpiresIn", 0)
        key = params["Key"]
        return (
            f"https://s3.test/{params['Bucket']}/{key}"
            f"?method={method}&expires={expires_in}&params={params}"
        )

    async def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        return self.objects[key]

    async def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key in self.delete_errors:
            raise ClientError(
                {"Error": {"Code": "500", "Message": "Error"}},
                "DeleteObject",
            )
        self.objects.pop(key, None)
        return {}


def configured_values() -> dict[str, str]:
    return {
        "ACCESS_KEY_ID": "key-id",
        "ACCESS_KEY_SECRET": "key-secret",
        "ENDPOINT": "https://minio.example.com",
        "REGION": "us-east-1",
        "BUCKET": "example-bucket",
    }


def make_provider(values: dict[str, str], client: FakeS3Client) -> S3StorageProvider:
    @asynccontextmanager
    async def factory():
        yield client

    return S3StorageProvider(config=FakeConfig(values), client_factory=factory)


async def test_provider_returns_none_when_incomplete() -> None:
    provider = make_provider({}, FakeS3Client())
    assert provider.get_storage() is None
    assert provider.get_storage("example-bucket") is None


async def test_provider_signs_put_and_get() -> None:
    provider = make_provider(configured_values(), FakeS3Client())
    storage = provider.get_storage()
    assert storage is not None

    signed_put = await storage.upload_async("storages/a", "text/plain", is_public=False)
    assert signed_put.method == "PUT"
    assert signed_put.headers == {
        "Content-Type": "text/plain",
        "x-amz-acl": "private",
    }
    assert "storages/a" in signed_put.url
    assert "expires=600" in signed_put.url

    public_put = await storage.upload_async("storages/b", "text/plain", is_public=True)
    assert public_put.headers["x-amz-acl"] == "public-read"

    signed_get = await storage.download_async(
        "storages/a",
        signed=True,
        content_type="text/plain",
        filename="demo.txt",
    )
    assert signed_get.method == "GET"
    assert signed_get.expires_at is not None
    assert "ResponseContentType" in signed_get.url
    assert "ResponseContentDisposition" in signed_get.url

    public_get = await storage.download_async(
        "storages/a",
        signed=False,
        content_type="text/plain",
        filename="demo.txt",
    )
    assert public_get.expires_at is None
    assert public_get.url == "https://minio.example.com/example-bucket/storages/a"


async def test_provider_maps_peek_and_delete() -> None:
    client = FakeS3Client()
    client.objects["storages/a"] = {
        "ContentLength": 12,
        "ContentType": "text/plain",
        "ETag": '"etag"',
    }
    provider = make_provider(configured_values(), client)
    storage = provider.get_storage("example-bucket")
    assert storage is not None

    head = await storage.peek_async("storages/a")
    assert head is not None
    assert (head.byte_size, head.content_type, head.etag) == (
        12,
        "text/plain",
        '"etag"',
    )
    assert await storage.peek_async("storages/missing") is None

    await storage.delete_async("storages/a")
    assert "storages/a" not in client.objects
    await storage.delete_async("storages/missing")

    client.objects["storages/fail"] = {"ContentLength": 1, "ContentType": "text/plain"}
    client.delete_errors.add("storages/fail")
    with pytest.raises(StorageError):
        await storage.delete_async("storages/fail")


async def test_get_storage_override_expires() -> None:
    provider = make_provider(configured_values(), FakeS3Client())
    storage = provider.get_storage(
        upload_expires_in=timedelta(seconds=120),
        download_expires_in=timedelta(seconds=60),
    )
    assert storage is not None
    signed_put = await storage.upload_async("storages/a", "text/plain")
    assert "expires=120" in signed_put.url


async def test_aws_style_public_url_without_endpoint() -> None:
    values = configured_values()
    values["ENDPOINT"] = ""
    provider = make_provider(values, FakeS3Client())
    storage = provider.get_storage("example-bucket")
    assert storage is not None
    link = await storage.download_async(
        "storages/a",
        signed=False,
        content_type="text/plain",
        filename="demo.txt",
    )
    assert link.url == "https://example-bucket.s3.us-east-1.amazonaws.com/storages/a"


async def test_key_prefix_is_applied_to_backend_keys() -> None:
    client = FakeS3Client()
    provider = make_provider(configured_values(), client)
    storage = provider.get_storage(key_prefix="/users/u1/")
    assert storage is not None
    signed = await storage.upload_async("file-uuid", "text/plain")
    assert "users/u1/file-uuid" in signed.url
    client.objects["users/u1/file-uuid"] = {
        "ContentLength": 1,
        "ContentType": "text/plain",
        "ETag": '"e"',
    }
    head = await storage.peek_async("file-uuid")
    assert head is not None
    await storage.delete_async("file-uuid")
    assert "users/u1/file-uuid" not in client.objects
