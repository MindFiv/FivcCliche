"""Create-time validation for user-scoped config APIs.

``user_*.api_key`` is NOT NULL and ids/urls have column limits, so bad input
must be a 422 rather than a database error (500). Duplicate ids in the same
scope are a 409; Postgres' unique index alone misses global (NULL user) rows.
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_api_client


@pytest.fixture
def client():
    yield from make_api_client(["users", "agent_configs"])


@pytest.fixture
def admin_headers(client: TestClient) -> dict[str, str]:
    token = client.post("/users/login", json={"username": "admin", "password": "admin123"}).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_headers(client: TestClient, admin_headers: dict[str, str]) -> dict[str, str]:
    client.post(
        "/users/",
        json={"username": "testuser", "email": "test@example.com", "password": "password123"},
        headers=admin_headers,
    )
    token = client.post(
        "/users/login", json={"username": "testuser", "password": "password123"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


KEYED_CONFIGS = {
    "/configs/embeddings/": {"provider": "openai", "model": "text-embedding-3-small"},
    "/configs/models/": {"provider": "openai", "model": "qwen-plus"},
    "/configs/asrs/": {"model": "paraformer-v2"},
    "/configs/tts/": {"model": "qwen-tts"},
}

ALL_CONFIGS = {
    **{url: {**body, "api_key": "sk-test"} for url, body in KEYED_CONFIGS.items()},
    "/configs/agents/": {"model_id": "default"},
    "/configs/tools/": {"description": "Tool", "transport": "stdio", "command": "python"},
    "/configs/skills/": {"description": "Skill", "instructions": "Do it."},
}


def _error_locs(response) -> list[tuple]:
    return [tuple(err["loc"]) for err in response.json()["detail"]]


@pytest.mark.parametrize("url", KEYED_CONFIGS)
@pytest.mark.parametrize("api_key", ["missing", None])
def test_create_without_api_key_is_422(client, admin_headers, url, api_key) -> None:
    body = {"id": "no-key", **KEYED_CONFIGS[url]}
    if api_key != "missing":
        body["api_key"] = api_key
    response = client.post(url, json=body, headers=admin_headers)
    assert response.status_code == 422
    assert ("body", "api_key") in _error_locs(response)


@pytest.mark.parametrize("url", KEYED_CONFIGS)
def test_create_with_empty_api_key_is_allowed(client, admin_headers, url) -> None:
    # Keyless providers (for example a local Ollama) send an empty string.
    body = {"id": "empty-key", **KEYED_CONFIGS[url], "api_key": ""}
    response = client.post(url, json=body, headers=admin_headers)
    assert response.status_code == 201, response.text
    assert "api_key" not in response.json()


@pytest.mark.parametrize("url", ALL_CONFIGS)
def test_create_with_too_long_id_is_422(client, admin_headers, url) -> None:
    response = client.post(url, json={"id": "x" * 37, **ALL_CONFIGS[url]}, headers=admin_headers)
    assert response.status_code == 422
    assert ("body", "id") in _error_locs(response)


@pytest.mark.parametrize("url", KEYED_CONFIGS)
@pytest.mark.parametrize("field", ["base_url", "api_key", "model"])
def test_create_with_too_long_field_is_422(client, admin_headers, url, field) -> None:
    body = {"id": "long-field", **ALL_CONFIGS[url], field: "x" * 256}
    response = client.post(url, json=body, headers=admin_headers)
    assert response.status_code == 422
    assert ("body", field) in _error_locs(response)


@pytest.mark.parametrize("url", KEYED_CONFIGS)
def test_update_with_too_long_base_url_is_422(client, admin_headers, url) -> None:
    created = client.post(url, json={"id": "patch-me", **ALL_CONFIGS[url]}, headers=admin_headers)
    assert created.status_code == 201, created.text
    response = client.patch(
        f"{url}{created.json()['uuid']}/", json={"base_url": "x" * 256}, headers=admin_headers
    )
    assert response.status_code == 422


@pytest.mark.parametrize("url", ALL_CONFIGS)
def test_duplicate_global_id_is_409(client, admin_headers, url) -> None:
    body = {"id": "dup", **ALL_CONFIGS[url]}
    assert client.post(url, json=body, headers=admin_headers).status_code == 201
    response = client.post(url, json=body, headers=admin_headers)
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
    assert client.get(url, headers=admin_headers).json()["total"] == 1


@pytest.mark.parametrize("url", ALL_CONFIGS)
def test_duplicate_user_id_is_409(client, user_headers, url) -> None:
    body = {"id": "dup", **ALL_CONFIGS[url]}
    assert client.post(url, json=body, headers=user_headers).status_code == 201
    assert client.post(url, json=body, headers=user_headers).status_code == 409


@pytest.mark.parametrize("url", ALL_CONFIGS)
def test_same_id_in_other_scope_is_allowed(client, admin_headers, user_headers, url) -> None:
    body = {"id": "shared", **ALL_CONFIGS[url]}
    assert client.post(url, json=body, headers=admin_headers).status_code == 201
    response = client.post(url, json=body, headers=user_headers)
    assert response.status_code == 201, response.text
    assert response.json()["user_uuid"] is not None
