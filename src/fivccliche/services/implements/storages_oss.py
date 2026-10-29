"""Aliyun OSS implementation of IStorageProvider / IStorage."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import oss2
from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs

from fivccliche.services.interfaces.storages import (
    IStorage,
    IStorageProvider,
    StorageDownLink,
    StorageError,
    StorageInfo,
    StorageUpLink,
)
from fivccliche.utils.types import to_optional_int, to_string

_DEFAULT_UPLOAD_EXPIRES = 600
_DEFAULT_DOWNLOAD_EXPIRES = 300


def _expires_seconds(value: timedelta | None, default: int) -> int:
    if value is None:
        return default
    seconds = int(value.total_seconds())
    return seconds if seconds > 0 else default


def _normalize_key_prefix(key_prefix: str | None) -> str:
    if not key_prefix:
        return ""
    return key_prefix.strip("/")


def _validate_relative_key(object_key: str) -> str:
    if not object_key or object_key.startswith("/") or ".." in object_key.split("/"):
        raise StorageError("Invalid storage object key")
    return object_key


class OssStorage(IStorage):
    """Object operations against one OSS bucket."""

    def __init__(
        self,
        bucket: Any,
        *,
        bucket_name: str,
        endpoint: str,
        upload_expires_in: int,
        download_expires_in: int,
        key_prefix: str = "",
    ) -> None:
        self._bucket = bucket
        self._bucket_name = bucket_name
        self._endpoint = endpoint.removeprefix("https://").removeprefix("http://")
        self._upload_expires_in = upload_expires_in
        self._download_expires_in = download_expires_in
        self._key_prefix = key_prefix

    def _full_key(self, object_key: str) -> str:
        relative = _validate_relative_key(object_key)
        if not self._key_prefix:
            return relative
        return f"{self._key_prefix}/{relative}"

    def _sign_put(self, object_key: str, content_type: str, *, is_public: bool) -> StorageUpLink:
        headers = {
            "Content-Type": content_type,
            "x-oss-object-acl": "public-read" if is_public else "private",
        }
        full_key = self._full_key(object_key)
        url = self._bucket.sign_url(
            "PUT",
            full_key,
            self._upload_expires_in,
            headers=headers,
            slash_safe=True,
        )
        return StorageUpLink(
            method="PUT",
            url=url,
            headers=headers,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._upload_expires_in),
        )

    def _sign_get(
        self,
        object_key: str,
        content_type: str,
        filename: str,
    ) -> StorageDownLink:
        quoted = quote(filename, safe="")
        params = {
            "response-content-type": content_type,
            "response-content-disposition": f"attachment; filename*=UTF-8''{quoted}",
        }
        full_key = self._full_key(object_key)
        url = self._bucket.sign_url(
            "GET",
            full_key,
            self._download_expires_in,
            params=params,
            slash_safe=True,
        )
        return StorageDownLink(
            method="GET",
            url=url,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._download_expires_in),
        )

    def _public_url(self, object_key: str) -> StorageDownLink:
        key = quote(self._full_key(object_key), safe="/")
        url = f"https://{self._bucket_name}.{self._endpoint}/{key}"
        return StorageDownLink(method="GET", url=url, expires_at=None)

    def _peek(self, object_key: str) -> StorageInfo | None:
        try:
            result = self._bucket.head_object(self._full_key(object_key))
        except oss2.exceptions.NoSuchKey:
            return None
        except oss2.exceptions.OssError as exc:
            raise StorageError("OSS HEAD failed for storage object") from exc
        return StorageInfo(
            byte_size=int(result.content_length),
            content_type=str(result.headers.get("Content-Type", "")),
            etag=result.headers.get("ETag"),
        )

    def _delete(self, object_key: str) -> None:
        try:
            self._bucket.delete_object(self._full_key(object_key))
        except oss2.exceptions.NoSuchKey:
            return
        except oss2.exceptions.OssError as exc:
            raise StorageError("OSS DELETE failed for storage object") from exc

    async def upload_async(
        self,
        object_key: str,
        content_type: str,
        *,
        is_public: bool = False,
    ) -> StorageUpLink:
        return await asyncio.to_thread(
            self._sign_put, object_key, content_type, is_public=is_public
        )

    async def download_async(
        self,
        object_key: str,
        *,
        signed: bool = True,
        content_type: str,
        filename: str,
    ) -> StorageDownLink:
        if not signed:
            return self._public_url(object_key)
        return await asyncio.to_thread(self._sign_get, object_key, content_type, filename)

    async def peek_async(self, object_key: str) -> StorageInfo | None:
        return await asyncio.to_thread(self._peek, object_key)

    async def delete_async(self, object_key: str) -> None:
        await asyncio.to_thread(self._delete, object_key)


class OssStorageProvider(IStorageProvider):
    """Factory over the official synchronous oss2 SDK."""

    def __init__(
        self,
        component_site: IComponentSite | None = None,
        *,
        config: configs.IConfig | None = None,
        bucket: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        if config is None and component_site is not None:
            config = query_component(component_site, configs.IConfig)
        self._config = config
        self._injected_bucket = bucket
        values = self._read_config()
        self._access_key_id = values["ACCESS_KEY_ID"]
        self._access_key_secret = values["ACCESS_KEY_SECRET"]
        self._endpoint = values["ENDPOINT"]
        self._default_bucket = values["BUCKET"]
        self._upload_expires = values["UPLOAD_URL_EXPIRES_SECONDS"]
        self._download_expires = values["DOWNLOAD_URL_EXPIRES_SECONDS"]

    def _read_config(self) -> dict[str, Any]:
        if self._config is None:
            return {
                "ACCESS_KEY_ID": "",
                "ACCESS_KEY_SECRET": "",
                "ENDPOINT": "",
                "BUCKET": "",
                "UPLOAD_URL_EXPIRES_SECONDS": _DEFAULT_UPLOAD_EXPIRES,
                "DOWNLOAD_URL_EXPIRES_SECONDS": _DEFAULT_DOWNLOAD_EXPIRES,
            }

        session = self._config.get_session("OSS")

        def text(key: str) -> str:
            return to_string(session.get_value(key), "")

        upload_expires = to_optional_int(session.get_value("UPLOAD_URL_EXPIRES_SECONDS"))
        download_expires = to_optional_int(session.get_value("DOWNLOAD_URL_EXPIRES_SECONDS"))
        return {
            "ACCESS_KEY_ID": text("ACCESS_KEY_ID"),
            "ACCESS_KEY_SECRET": text("ACCESS_KEY_SECRET"),
            "ENDPOINT": text("ENDPOINT"),
            "BUCKET": text("BUCKET"),
            "UPLOAD_URL_EXPIRES_SECONDS": (
                upload_expires
                if upload_expires is not None and upload_expires > 0
                else _DEFAULT_UPLOAD_EXPIRES
            ),
            "DOWNLOAD_URL_EXPIRES_SECONDS": (
                download_expires
                if download_expires is not None and download_expires > 0
                else _DEFAULT_DOWNLOAD_EXPIRES
            ),
        }

    def get_storage(
        self,
        name: str | None = None,
        *,
        key_prefix: str | None = None,
        upload_expires_in: timedelta | None = None,
        download_expires_in: timedelta | None = None,
        **kwargs: Any,
    ) -> IStorage | None:
        bucket_name = self._default_bucket if name is None else name
        if not bucket_name or not all(
            (self._access_key_id, self._access_key_secret, self._endpoint)
        ):
            return None

        upload_expires = _expires_seconds(upload_expires_in, int(self._upload_expires))
        download_expires = _expires_seconds(download_expires_in, int(self._download_expires))
        prefix = key_prefix if key_prefix is not None else kwargs.get("key_prefix")

        if self._injected_bucket is not None:
            bucket = self._injected_bucket
        else:
            auth = oss2.Auth(self._access_key_id, self._access_key_secret)
            bucket = oss2.Bucket(auth, self._endpoint, bucket_name)

        return OssStorage(
            bucket,
            bucket_name=bucket_name,
            endpoint=self._endpoint,
            upload_expires_in=upload_expires,
            download_expires_in=download_expires,
            key_prefix=_normalize_key_prefix(prefix if isinstance(prefix, str) else None),
        )
