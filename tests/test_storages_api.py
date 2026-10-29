"""API tests for user-owned storage files and folders."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fivccliche.modules.storages.models import StorageFile
from fivccliche.services.interfaces.storages import (
    IStorage,
    IStorageProvider,
    StorageDownLink,
    StorageError,
    StorageInfo,
    StorageUpLink,
)
from fivccliche.utils.deps import get_storage_provider_async
from tests.conftest import make_api_client


class FakeStorage(IStorage):
    def __init__(self) -> None:
        self.objects: dict[str, StorageInfo] = {}
        self.deleted: list[str] = []
        self.delete_failures: set[str] = set()
        self.last_keys: list[str] = []

    async def upload_async(
        self,
        object_key: str,
        content_type: str,
        *,
        is_public: bool = False,
    ) -> StorageUpLink:
        self.last_keys.append(object_key)
        headers = {
            "Content-Type": content_type,
            "x-oss-object-acl": "public-read" if is_public else "private",
        }
        return StorageUpLink(
            method="PUT",
            url=f"https://oss.test/{object_key}",
            headers=headers,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
        )

    async def download_async(
        self,
        object_key: str,
        *,
        signed: bool = True,
        content_type: str,
        filename: str,
    ) -> StorageDownLink:
        self.last_keys.append(object_key)
        if not signed:
            return StorageDownLink(
                method="GET",
                url=f"https://oss.test/{object_key}?public=1",
                expires_at=None,
            )
        return StorageDownLink(
            method="GET",
            url=f"https://oss.test/{object_key}?download={filename}",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=300),
        )

    async def peek_async(self, object_key: str) -> StorageInfo | None:
        self.last_keys.append(object_key)
        return self.objects.get(object_key)

    async def delete_async(self, object_key: str) -> None:
        self.last_keys.append(object_key)
        if object_key in self.delete_failures:
            raise StorageError("delete failed")
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)


class PrefixedFakeStorage(IStorage):
    """Applies key_prefix like S3Storage/OssStorage for cleanup / prefix assertions."""

    def __init__(self, inner: FakeStorage, key_prefix: str = "") -> None:
        self._inner = inner
        self._key_prefix = key_prefix.strip("/")

    def _full_key(self, object_key: str) -> str:
        if not self._key_prefix:
            return object_key
        return f"{self._key_prefix}/{object_key}"

    async def upload_async(
        self,
        object_key: str,
        content_type: str,
        *,
        is_public: bool = False,
    ) -> StorageUpLink:
        return await self._inner.upload_async(
            self._full_key(object_key), content_type, is_public=is_public
        )

    async def download_async(
        self,
        object_key: str,
        *,
        signed: bool = True,
        content_type: str,
        filename: str,
    ) -> StorageDownLink:
        return await self._inner.download_async(
            self._full_key(object_key),
            signed=signed,
            content_type=content_type,
            filename=filename,
        )

    async def peek_async(self, object_key: str) -> StorageInfo | None:
        return await self._inner.peek_async(self._full_key(object_key))

    async def delete_async(self, object_key: str) -> None:
        await self._inner.delete_async(self._full_key(object_key))


class FakeStorageProvider(IStorageProvider):
    def __init__(self, storage: FakeStorage | None = None) -> None:
        self.storage = storage or FakeStorage()
        self.calls: list[tuple[str | None, str | None]] = []

    def get_storage(
        self,
        name: str | None = None,
        *,
        key_prefix: str | None = None,
        upload_expires_in: timedelta | None = None,
        download_expires_in: timedelta | None = None,
        **kwargs: Any,
    ) -> IStorage | None:
        if name is not None and not name:
            return None
        self.calls.append((name, key_prefix))
        prefix = (key_prefix or "").strip("/")
        return PrefixedFakeStorage(self.storage, prefix)


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def storage_provider(storage: FakeStorage) -> FakeStorageProvider:
    return FakeStorageProvider(storage)


@pytest.fixture
def client(storage_provider: FakeStorageProvider):
    for test_client in make_api_client(["users", "storages"]):
        test_client.app.dependency_overrides[get_storage_provider_async] = lambda: storage_provider
        yield test_client


@pytest.fixture
def auth_token(client: TestClient) -> str:
    response = client.post("/users/login", json={"username": "admin", "password": "admin123"})
    return response.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_upload(
    client: TestClient,
    token: str,
    *,
    filename: str = "demo.txt",
    content_type: str = "text/plain",
    byte_size: int = 12,
    is_public: bool = False,
    folder_uuid: str | None = None,
) -> Any:
    body: dict[str, Any] = {
        "filename": filename,
        "content_type": content_type,
        "byte_size": byte_size,
        "is_public": is_public,
    }
    if folder_uuid is not None:
        body["folder_uuid"] = folder_uuid
    return client.post(
        "/files/uploads/",
        headers=auth_headers(token),
        json=body,
    )


def test_create_upload_returns_pending_metadata(
    client: TestClient,
    auth_token: str,
    storage_provider: FakeStorageProvider,
) -> None:
    response = create_upload(client, auth_token)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "PENDING"
    assert data["user_uuid"] is None  # admin / superuser creates globals
    assert data["folder_uuid"] is None
    assert data["is_public"] is False
    assert data["upload_method"] == "PUT"
    assert data["upload_headers"] == {
        "Content-Type": "text/plain",
        "x-oss-object-acl": "private",
    }
    assert data["object_key"] == data["uuid"]
    assert storage_provider.calls[-1] == (None, "")


def test_create_public_upload_includes_acl_header(
    client: TestClient,
    auth_token: str,
) -> None:
    response = create_upload(client, auth_token, is_public=True)
    assert response.status_code == 201
    assert response.json()["is_public"] is True
    assert response.json()["upload_headers"]["x-oss-object-acl"] == "public-read"


def test_create_upload_rejects_disallowed_type(
    client: TestClient,
    auth_token: str,
) -> None:
    response = create_upload(
        client,
        auth_token,
        filename="demo.exe",
        content_type="application/octet-stream",
    )

    assert response.status_code == 422


def test_regular_user_default_prefix(
    client: TestClient,
    storage: FakeStorage,
    storage_provider: FakeStorageProvider,
) -> None:
    token = create_user_token(client, username="prefixuser", email="prefix@example.com")
    response = create_upload(client, token)
    assert response.status_code == 201
    data = response.json()
    assert data["user_uuid"] is not None
    assert storage_provider.calls[-1] == (None, f"users/{data['user_uuid']}")
    assert storage.last_keys[-1] == f"users/{data['user_uuid']}/{data['object_key']}"


def test_duplicate_filename_allowed(
    client: TestClient,
    auth_token: str,
) -> None:
    first = create_upload(client, auth_token, filename="same.txt")
    second = create_upload(client, auth_token, filename="same.txt")
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["uuid"] != second.json()["uuid"]


def test_folder_crud_superuser_only(
    client: TestClient,
    auth_token: str,
) -> None:
    regular = create_user_token(client, username="nofolder", email="nofolder@example.com")
    denied = client.post(
        "/folders/",
        headers=auth_headers(regular),
        json={"name": "app", "bucket_name": "b1", "bucket_prefix": "/uploads/app"},
    )
    assert denied.status_code == 403

    created = client.post(
        "/folders/",
        headers=auth_headers(auth_token),
        json={"name": "app", "bucket_name": "b1", "bucket_prefix": "/uploads/app"},
    )
    assert created.status_code == 201
    folder = created.json()
    assert folder["user_uuid"] is None
    assert folder["bucket_prefix"] == "uploads/app"

    listed = client.get("/folders/", headers=auth_headers(regular))
    assert listed.status_code == 200
    assert folder["uuid"] in [item["uuid"] for item in listed.json()["results"]]

    patched = client.patch(
        f"/folders/{folder['uuid']}/",
        headers=auth_headers(auth_token),
        json={"bucket_prefix": "uploads/app2"},
    )
    assert patched.status_code == 200
    assert patched.json()["bucket_prefix"] == "uploads/app2"

    assert (
        client.delete(
            f"/folders/{folder['uuid']}/",
            headers=auth_headers(regular),
        ).status_code
        == 403
    )
    assert (
        client.delete(
            f"/folders/{folder['uuid']}/",
            headers=auth_headers(auth_token),
        ).status_code
        == 204
    )


def test_upload_with_folder_uses_folder_storage(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
    storage_provider: FakeStorageProvider,
) -> None:
    folder = client.post(
        "/folders/",
        headers=auth_headers(auth_token),
        json={"name": "media", "bucket_name": "media-bucket", "bucket_prefix": "media"},
    ).json()
    response = create_upload(client, auth_token, folder_uuid=folder["uuid"])
    assert response.status_code == 201
    data = response.json()
    assert data["folder_uuid"] == folder["uuid"]
    assert storage_provider.calls[-1] == ("media-bucket", "media")
    assert storage.last_keys[-1] == f"media/{data['object_key']}"


def test_complete_upload_verifies_object(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    created = create_upload(client, auth_token).json()
    # global file → empty prefix; PrefixedFakeStorage uses relative key as-is
    storage.objects[created["object_key"]] = StorageInfo(
        byte_size=12,
        content_type="text/plain",
        etag='"etag"',
    )

    response = client.post(
        f"/files/{created['uuid']}/complete/",
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert data["etag"] == '"etag"'

    repeat = client.post(
        f"/files/{created['uuid']}/complete/",
        headers=auth_headers(auth_token),
    )
    assert repeat.status_code == 200
    assert repeat.json()["status"] == "READY"


def test_complete_upload_rejects_missing_object(
    client: TestClient,
    auth_token: str,
) -> None:
    created = create_upload(client, auth_token).json()

    response = client.post(
        f"/files/{created['uuid']}/complete/",
        headers=auth_headers(auth_token),
    )

    assert response.status_code == 400


def test_complete_upload_rejects_mismatch_and_reclaims_object(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    created = create_upload(client, auth_token, byte_size=12).json()
    storage.objects[created["object_key"]] = StorageInfo(
        byte_size=13,
        content_type="text/plain",
        etag=None,
    )

    response = client.post(
        f"/files/{created['uuid']}/complete/",
        headers=auth_headers(auth_token),
    )

    assert response.status_code == 400
    assert created["object_key"] in storage.deleted
    row = client.loop.run_until_complete(client.async_session.get(StorageFile, created["uuid"]))
    assert row is None


def create_ready_file(
    client: TestClient,
    token: str,
    storage: FakeStorage,
    *,
    is_public: bool = False,
    folder_uuid: str | None = None,
    key_prefix: str | None = None,
) -> dict[str, Any]:
    created = create_upload(client, token, is_public=is_public, folder_uuid=folder_uuid).json()
    if key_prefix is None:
        prefix = f"users/{created['user_uuid']}" if created["user_uuid"] else ""
    else:
        prefix = key_prefix.strip("/")
    full_key = f"{prefix}/{created['object_key']}" if prefix else created["object_key"]
    storage.objects[full_key] = StorageInfo(
        byte_size=12,
        content_type="text/plain",
        etag='"etag"',
    )
    completed = client.post(
        f"/files/{created['uuid']}/complete/",
        headers=auth_headers(token),
    )
    assert completed.status_code == 200
    return completed.json()


def test_list_defaults_to_ready_files(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    pending = create_upload(client, auth_token).json()
    ready = create_ready_file(client, auth_token, storage)

    response = client.get("/files/", headers=auth_headers(auth_token))

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert [item["uuid"] for item in data["results"]] == [ready["uuid"]]

    pending_list = client.get(
        "/files/",
        headers=auth_headers(auth_token),
        params={"status": "PENDING"},
    )
    assert [item["uuid"] for item in pending_list.json()["results"]] == [pending["uuid"]]


def test_list_is_global_filter(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    global_file = create_ready_file(client, auth_token, storage)
    owner_token = create_user_token(client, username="sharedowner", email="sharedowner@example.com")
    owned = create_ready_file(client, owner_token, storage)
    assert owned["user_uuid"] is not None

    global_only = client.get(
        "/files/",
        headers=auth_headers(auth_token),
        params={"is_global": True},
    )
    assert global_only.status_code == 200
    assert all(item["user_uuid"] is None for item in global_only.json()["results"])
    assert global_file["uuid"] in [item["uuid"] for item in global_only.json()["results"]]

    not_global = client.get(
        "/files/",
        headers=auth_headers(owner_token),
        params={"is_global": False},
    )
    assert not_global.status_code == 200
    assert all(item["user_uuid"] is not None for item in not_global.json()["results"])
    assert owned["uuid"] in [item["uuid"] for item in not_global.json()["results"]]


def test_private_url_requires_owner_and_ready(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    pending = create_upload(client, auth_token).json()
    conflict = client.get(
        f"/files/{pending['uuid']}/url/",
        headers=auth_headers(auth_token),
    )
    assert conflict.status_code == 409

    ready = create_ready_file(client, auth_token, storage)
    response = client.get(
        f"/files/{ready['uuid']}/url/",
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["url"].startswith("https://oss.test/")
    assert body["expires_at"] is not None


def create_user_token(
    client: TestClient,
    *,
    username: str,
    email: str,
) -> str:
    admin_token = client.post(
        "/users/login",
        json={"username": "admin", "password": "admin123"},
    ).json()["access_token"]
    client.post(
        "/users/",
        headers=auth_headers(admin_token),
        json={
            "username": username,
            "email": email,
            "password": "password123",
        },
    )
    return client.post(
        "/users/login",
        json={"username": username, "password": "password123"},
    ).json()["access_token"]


def test_owned_url_rejects_another_user(
    client: TestClient,
    storage: FakeStorage,
) -> None:
    owner_token = create_user_token(client, username="fileowner", email="owner@example.com")
    other_token = create_user_token(client, username="otheruser", email="other@example.com")
    ready = create_ready_file(client, owner_token, storage)
    assert ready["user_uuid"] is not None

    response = client.get(
        f"/files/{ready['uuid']}/url/",
        headers=auth_headers(other_token),
    )

    assert response.status_code == 404


def test_global_file_readable_but_not_writable_by_regular_user(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    pending = create_upload(client, auth_token).json()
    assert pending["user_uuid"] is None
    storage.objects[pending["object_key"]] = StorageInfo(
        byte_size=12,
        content_type="text/plain",
        etag='"etag"',
    )
    regular_token = create_user_token(client, username="regular", email="regular@example.com")

    assert (
        client.post(
            f"/files/{pending['uuid']}/complete/",
            headers=auth_headers(regular_token),
        ).status_code
        == 404
    )

    ready = client.post(
        f"/files/{pending['uuid']}/complete/",
        headers=auth_headers(auth_token),
    ).json()
    assert ready["status"] == "READY"

    listed = client.get("/files/", headers=auth_headers(regular_token))
    assert ready["uuid"] in [item["uuid"] for item in listed.json()["results"]]

    url = client.get(
        f"/files/{ready['uuid']}/url/",
        headers=auth_headers(regular_token),
    )
    assert url.status_code == 200

    assert (
        client.delete(
            f"/files/{ready['uuid']}/",
            headers=auth_headers(regular_token),
        ).status_code
        == 404
    )

    assert (
        client.delete(
            f"/files/{ready['uuid']}/",
            headers=auth_headers(auth_token),
        ).status_code
        == 202
    )


def test_public_url_returns_unsigned_link(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    public = create_ready_file(client, auth_token, storage, is_public=True)
    assert public["is_public"] is True

    response = client.get(
        f"/files/{public['uuid']}/url/",
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["url"].endswith("?public=1")
    assert body["expires_at"] is None


def test_is_public_has_no_patch_route(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    ready = create_ready_file(client, auth_token, storage)
    response = client.patch(
        f"/files/{ready['uuid']}/",
        headers=auth_headers(auth_token),
        json={"is_public": True},
    )
    assert response.status_code == 405


def test_delete_claims_file_and_removes_object(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    ready = create_ready_file(client, auth_token, storage)

    response = client.delete(
        f"/files/{ready['uuid']}/",
        headers=auth_headers(auth_token),
    )

    assert response.status_code == 202
    assert ready["object_key"] in storage.deleted
    assert (
        client.get(
            f"/files/{ready['uuid']}/",
            headers=auth_headers(auth_token),
        ).status_code
        == 404
    )


def test_delete_keeps_row_for_cleanup_when_oss_fails(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    ready = create_ready_file(client, auth_token, storage)
    storage.delete_failures.add(ready["object_key"])

    response = client.delete(
        f"/files/{ready['uuid']}/",
        headers=auth_headers(auth_token),
    )
    detail = client.get(
        f"/files/{ready['uuid']}/",
        headers=auth_headers(auth_token),
    )

    assert response.status_code == 202
    assert detail.status_code == 200
    assert detail.json()["status"] == "DELETING"
