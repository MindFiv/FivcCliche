"""API tests for user-owned OSS storage files."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fivccliche.modules.storages.models import StorageFile
from fivccliche.modules.storages.routers import get_required_storage_async
from fivccliche.services.interfaces.storages import (
    IStorage,
    IStorageProvider,
    StorageDownLink,
    StorageError,
    StorageInfo,
    StorageUpLink,
)
from tests.conftest import make_api_client


class FakeStorage(IStorage):
    def __init__(self) -> None:
        self.objects: dict[str, StorageInfo] = {}
        self.deleted: list[str] = []
        self.delete_failures: set[str] = set()

    async def upload_async(
        self,
        object_key: str,
        content_type: str,
        *,
        is_public: bool = False,
    ) -> StorageUpLink:
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
        return self.objects.get(object_key)

    async def delete_async(self, object_key: str) -> None:
        if object_key in self.delete_failures:
            raise StorageError("delete failed")
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)


class FakeStorageProvider(IStorageProvider):
    def __init__(self, storage: FakeStorage | None = None) -> None:
        self.storage = storage or FakeStorage()

    def get_storage(
        self,
        name: str | None = None,
        *,
        upload_expires_in: timedelta | None = None,
        download_expires_in: timedelta | None = None,
        **kwargs: Any,
    ) -> IStorage | None:
        if name is not None and not name:
            return None
        return self.storage


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def storage_provider(storage: FakeStorage) -> FakeStorageProvider:
    return FakeStorageProvider(storage)


@pytest.fixture
def client(storage: FakeStorage):
    for test_client in make_api_client(["users", "storages"]):
        test_client.app.dependency_overrides[get_required_storage_async] = lambda: storage
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
) -> Any:
    return client.post(
        "/files/uploads/",
        headers=auth_headers(token),
        json={
            "filename": filename,
            "content_type": content_type,
            "byte_size": byte_size,
            "is_public": is_public,
        },
    )


def test_create_upload_returns_pending_metadata(
    client: TestClient,
    auth_token: str,
) -> None:
    response = create_upload(client, auth_token)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "PENDING"
    assert data["user_uuid"] is None  # admin / superuser creates globals
    assert data["is_public"] is False
    assert data["upload_method"] == "PUT"
    assert data["upload_headers"] == {
        "Content-Type": "text/plain",
        "x-oss-object-acl": "private",
    }
    assert data["object_key"].startswith("storages/")


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


def test_complete_upload_verifies_object(
    client: TestClient,
    auth_token: str,
    storage: FakeStorage,
) -> None:
    created = create_upload(client, auth_token).json()
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
) -> dict[str, Any]:
    created = create_upload(client, token, is_public=is_public).json()
    storage.objects[created["object_key"]] = StorageInfo(
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
