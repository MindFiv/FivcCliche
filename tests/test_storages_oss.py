"""Tests for the Aliyun OSS storage provider boundary."""

from datetime import timedelta
from types import SimpleNamespace

import oss2
import pytest

from fivccliche.services.implements.storages_oss import OssStorageProvider
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
        return FakeConfigSession(self.values if name == "OSS" else {})


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, SimpleNamespace] = {}
        self.delete_errors: set[str] = set()

    def sign_url(
        self,
        method: str,
        key: str,
        expires: int,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        slash_safe: bool = False,
    ) -> str:
        return (
            f"https://oss.test/{key}?method={method}&expires={expires}"
            f"&headers={headers}%26params={params}"
        )

    def head_object(self, key: str) -> SimpleNamespace:
        if key not in self.objects:
            raise oss2.exceptions.NoSuchKey(404, {}, b"", {})
        return self.objects[key]

    def delete_object(self, key: str) -> None:
        if key in self.delete_errors:
            raise oss2.exceptions.ServerError(500, {}, b"", {})
        if key not in self.objects:
            raise oss2.exceptions.NoSuchKey(404, {}, b"", {})
        self.objects.pop(key, None)


def configured_values() -> dict[str, str]:
    return {
        "ACCESS_KEY_ID": "key-id",
        "ACCESS_KEY_SECRET": "key-secret",
        "ENDPOINT": "oss-cn-hangzhou.aliyuncs.com",
        "BUCKET": "example-bucket",
    }


def make_provider(values: dict[str, str], bucket: FakeBucket) -> OssStorageProvider:
    return OssStorageProvider(config=FakeConfig(values), bucket=bucket)


async def test_provider_returns_none_when_incomplete() -> None:
    provider = make_provider({}, FakeBucket())

    assert provider.get_storage() is None
    assert provider.get_storage("example-bucket") is None


async def test_provider_signs_put_and_get() -> None:
    provider = make_provider(configured_values(), FakeBucket())
    storage = provider.get_storage()
    assert storage is not None

    signed_put = await storage.upload_async("storages/a", "text/plain", is_public=False)
    assert signed_put.method == "PUT"
    assert signed_put.headers == {
        "Content-Type": "text/plain",
        "x-oss-object-acl": "private",
    }
    assert "storages/a" in signed_put.url
    assert "expires=600" in signed_put.url

    public_put = await storage.upload_async("storages/b", "text/plain", is_public=True)
    assert public_put.headers["x-oss-object-acl"] == "public-read"

    signed_get = await storage.download_async(
        "storages/a",
        signed=True,
        content_type="text/plain",
        filename="demo.txt",
    )
    assert signed_get.method == "GET"
    assert signed_get.expires_at is not None
    assert "response-content-type" in signed_get.url
    assert "response-content-disposition" in signed_get.url

    public_get = await storage.download_async(
        "storages/a",
        signed=False,
        content_type="text/plain",
        filename="demo.txt",
    )
    assert public_get.expires_at is None
    assert public_get.url == ("https://example-bucket.oss-cn-hangzhou.aliyuncs.com/storages/a")


async def test_provider_maps_peek_and_delete() -> None:
    bucket = FakeBucket()
    bucket.objects["storages/a"] = SimpleNamespace(
        content_length=12,
        headers={"Content-Type": "text/plain", "ETag": '"etag"'},
    )
    provider = make_provider(configured_values(), bucket)
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
    assert "storages/a" not in bucket.objects
    await storage.delete_async("storages/missing")

    bucket.objects["storages/fail"] = SimpleNamespace(
        content_length=12,
        headers={},
    )
    bucket.delete_errors.add("storages/fail")
    with pytest.raises(StorageError):
        await storage.delete_async("storages/fail")


async def test_get_storage_override_expires() -> None:
    provider = make_provider(configured_values(), FakeBucket())
    storage = provider.get_storage(
        upload_expires_in=timedelta(seconds=120),
        download_expires_in=timedelta(seconds=60),
    )
    assert storage is not None
    signed_put = await storage.upload_async("storages/a", "text/plain")
    assert "expires=120" in signed_put.url


async def test_key_prefix_is_applied_to_backend_keys() -> None:
    bucket = FakeBucket()
    provider = make_provider(configured_values(), bucket)
    storage = provider.get_storage(key_prefix="/users/u1/")
    assert storage is not None
    signed = await storage.upload_async("file-uuid", "text/plain")
    assert "users/u1/file-uuid" in signed.url
    bucket.objects["users/u1/file-uuid"] = SimpleNamespace(
        content_length=1,
        headers={"Content-Type": "text/plain", "ETag": '"e"'},
    )
    head = await storage.peek_async("file-uuid")
    assert head is not None
    await storage.delete_async("file-uuid")
    assert "users/u1/file-uuid" not in bucket.objects
