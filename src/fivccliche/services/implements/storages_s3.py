"""S3 / S3-compatible implementation of IStorageProvider / IStorage via aioboto3."""

import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import aioboto3
from botocore.exceptions import ClientError
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
_DEFAULT_REGION = "us-east-1"

ClientFactory = Callable[..., AbstractAsyncContextManager[Any]]


def _expires_seconds(value: timedelta | None, default: int) -> int:
    if value is None:
        return default
    seconds = int(value.total_seconds())
    return seconds if seconds > 0 else default


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class S3Storage(IStorage):
    """Object operations against one S3 or S3-compatible bucket."""

    def __init__(
        self,
        *,
        bucket_name: str,
        region: str,
        endpoint: str,
        upload_expires_in: int,
        download_expires_in: int,
        client_factory: ClientFactory,
    ) -> None:
        self._bucket_name = bucket_name
        self._region = region
        self._endpoint = endpoint.rstrip("/")
        self._upload_expires_in = upload_expires_in
        self._download_expires_in = download_expires_in
        self._client_factory = client_factory

    def _public_url(self, object_key: str) -> StorageDownLink:
        key = quote(object_key, safe="/")
        if self._endpoint:
            base = self._endpoint
            if not base.startswith(("http://", "https://")):
                base = f"https://{base}"
            url = f"{base}/{self._bucket_name}/{key}"
        else:
            url = f"https://{self._bucket_name}.s3.{self._region}.amazonaws.com/{key}"
        return StorageDownLink(method="GET", url=url, expires_at=None)

    async def upload_async(
        self,
        object_key: str,
        content_type: str,
        *,
        is_public: bool = False,
    ) -> StorageUpLink:
        acl = "public-read" if is_public else "private"
        headers = {
            "Content-Type": content_type,
            "x-amz-acl": acl,
        }
        async with self._client_factory() as client:
            url = await _maybe_await(
                client.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": self._bucket_name,
                        "Key": object_key,
                        "ContentType": content_type,
                        "ACL": acl,
                    },
                    ExpiresIn=self._upload_expires_in,
                )
            )
        return StorageUpLink(
            method="PUT",
            url=url,
            headers=headers,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._upload_expires_in),
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
        quoted = quote(filename, safe="")
        async with self._client_factory() as client:
            url = await _maybe_await(
                client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self._bucket_name,
                        "Key": object_key,
                        "ResponseContentType": content_type,
                        "ResponseContentDisposition": f"attachment; filename*=UTF-8''{quoted}",
                    },
                    ExpiresIn=self._download_expires_in,
                )
            )
        return StorageDownLink(
            method="GET",
            url=url,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._download_expires_in),
        )

    async def peek_async(self, object_key: str) -> StorageInfo | None:
        try:
            async with self._client_factory() as client:
                result = await client.head_object(Bucket=self._bucket_name, Key=object_key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise StorageError("S3 HEAD failed for storage object") from exc
        return StorageInfo(
            byte_size=int(result["ContentLength"]),
            content_type=str(result.get("ContentType") or ""),
            etag=result.get("ETag"),
        )

    async def delete_async(self, object_key: str) -> None:
        try:
            async with self._client_factory() as client:
                await client.delete_object(Bucket=self._bucket_name, Key=object_key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return
            raise StorageError("S3 DELETE failed for storage object") from exc


class S3StorageProvider(IStorageProvider):
    """Factory over aioboto3 for AWS S3 and S3-compatible endpoints."""

    def __init__(
        self,
        component_site: IComponentSite | None = None,
        *,
        config: configs.IConfig | None = None,
        client_factory: ClientFactory | None = None,
        **_kwargs: Any,
    ) -> None:
        if config is None and component_site is not None:
            config = query_component(component_site, configs.IConfig)
        self._config = config
        self._injected_client_factory = client_factory
        values = self._read_config()
        self._access_key_id = values["ACCESS_KEY_ID"]
        self._access_key_secret = values["ACCESS_KEY_SECRET"]
        self._endpoint = values["ENDPOINT"]
        self._region = values["REGION"]
        self._default_bucket = values["BUCKET"]
        self._upload_expires = values["UPLOAD_URL_EXPIRES_SECONDS"]
        self._download_expires = values["DOWNLOAD_URL_EXPIRES_SECONDS"]
        self._session = aioboto3.Session()

    def _read_config(self) -> dict[str, Any]:
        if self._config is None:
            return {
                "ACCESS_KEY_ID": "",
                "ACCESS_KEY_SECRET": "",
                "ENDPOINT": "",
                "REGION": _DEFAULT_REGION,
                "BUCKET": "",
                "UPLOAD_URL_EXPIRES_SECONDS": _DEFAULT_UPLOAD_EXPIRES,
                "DOWNLOAD_URL_EXPIRES_SECONDS": _DEFAULT_DOWNLOAD_EXPIRES,
            }

        session = self._config.get_session("S3")

        def text(key: str) -> str:
            return to_string(session.get_value(key), "")

        upload_expires = to_optional_int(session.get_value("UPLOAD_URL_EXPIRES_SECONDS"))
        download_expires = to_optional_int(session.get_value("DOWNLOAD_URL_EXPIRES_SECONDS"))
        region = text("REGION") or _DEFAULT_REGION
        return {
            "ACCESS_KEY_ID": text("ACCESS_KEY_ID"),
            "ACCESS_KEY_SECRET": text("ACCESS_KEY_SECRET"),
            "ENDPOINT": text("ENDPOINT"),
            "REGION": region,
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

    def _make_client_factory(self) -> ClientFactory:
        if self._injected_client_factory is not None:
            return self._injected_client_factory

        endpoint = self._endpoint or None
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"

        @asynccontextmanager
        async def factory() -> AsyncIterator[Any]:
            async with self._session.client(
                "s3",
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._access_key_secret,
                region_name=self._region,
                endpoint_url=endpoint,
            ) as client:
                yield client

        return factory

    def get_storage(
        self,
        name: str | None = None,
        *,
        upload_expires_in: timedelta | None = None,
        download_expires_in: timedelta | None = None,
        **kwargs: Any,
    ) -> IStorage | None:
        bucket_name = self._default_bucket if name is None else name
        if not bucket_name or not all((self._access_key_id, self._access_key_secret)):
            return None

        upload_expires = _expires_seconds(upload_expires_in, int(self._upload_expires))
        download_expires = _expires_seconds(download_expires_in, int(self._download_expires))

        return S3Storage(
            bucket_name=bucket_name,
            region=self._region,
            endpoint=self._endpoint,
            upload_expires_in=upload_expires,
            download_expires_in=download_expires,
            client_factory=self._make_client_factory(),
        )
