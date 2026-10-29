# Storages and Aliyun OSS Design

## Summary

Add authenticated user file uploads backed by Aliyun OSS. The backend creates
file metadata and issues a short-lived presigned OSS upload URL, but file bytes
are uploaded directly from the client to OSS. The client then calls a completion
endpoint so the backend can verify the object with OSS before exposing it.

The module is named `storages`. HTTP resources remain file-oriented and are
mounted under `/api/files`.

## Goals

- Let an authenticated user upload a file directly to Aliyun OSS.
- Keep OSS object ACLs private for every file.
- Record file ownership, object location, status, declared type, size, and
  access permission in PostgreSQL.
- Support private owner access and immutable opt-in internet-public access.
- Verify uploaded objects with OSS `HEADObject` before exposing them.
- Reclaim pending uploads whose clients never call the completion endpoint.
- Reuse the same deletion workflow for pending, rejected, and ready files.

## Non-Goals

- No proxy upload through the FastAPI process.
- No multipart or resumable large-file upload.
- No virus scanning or semantic media validation.
- No OSS callback integration.
- No mutable file permission.
- No file search, sharing grants, folders, tags, comments, or versioning.
- No migration of objects that already exist outside this module.

## Public Access Semantics

`permission` is selected when the upload is created and is immutable:

- `PRIVATE`: only the owner can request a short-lived download URL.
- `PUBLIC_READ`: any unauthenticated client that knows the file UUID can access
  the public endpoint.

The OSS object ACL remains private in both cases. `PUBLIC_READ` does not modify
OSS or Bucket ACLs. It only allows the backend public route to verify the DB
state and redirect to a short-lived presigned GET URL.

There is no permission PATCH route. If a user selects the wrong permission, the
file must be deleted and uploaded again. This avoids accidental exposure
revocation issues, cached redirects, and ambiguity about already-issued GET
URLs after a state change.

Public responses set `Cache-Control: no-store`. Presigned GET URLs remain short
lived, with a default of 5 minutes.

## HTTP API

All authenticated routes use the normal Bearer token dependency.

### Create An Upload

`POST /api/files/uploads/`

Request:

```json
{
  "filename": "demo.png",
  "content_type": "image/png",
  "byte_size": 12345,
  "permission": "PRIVATE"
}
```

Validation:

- `filename` is required, has a maximum length of 255, and must be a single path
  component. It cannot contain `/`, a backslash, or a NUL character.
- `content_type` is required and must exactly match the MIME type allowed for
  the filename suffix.
- `byte_size` must be greater than zero and no greater than 20 MiB
  (`20 * 1024 * 1024`).
- `permission` must be `PRIVATE` or `PUBLIC_READ`.

Allowed type pairs:

| Kind | Suffix | MIME type |
|---|---|---|
| Image | `.gif` | `image/gif` |
| Image | `.jpg`, `.jpeg` | `image/jpeg` |
| Image | `.png` | `image/png` |
| Image | `.webp` | `image/webp` |
| PDF | `.pdf` | `application/pdf` |
| Text | `.csv` | `text/csv` |
| Text | `.json` | `application/json` |
| Text | `.md` | `text/markdown` |
| Text | `.txt` | `text/plain` |
| Audio | `.m4a` | `audio/mp4` |
| Audio | `.mp3` | `audio/mpeg` |
| Audio | `.wav` | `audio/wav` |
| Audio | `.weba` | `audio/webm` |
| Video | `.mov` | `video/quicktime` |
| Video | `.mp4` | `video/mp4` |
| Video | `.webm` | `video/webm` |

The handler first creates a `PENDING` row and commits it. It then creates a
presigned PUT URL for a fresh random object key, such as `storages/{uuid}`.
Signing happens after the DB commit so an upload URL is never issued without an
associated reclaimable database record.

The response is `201` and contains:

- File metadata, including UUID, filename, content type, byte size, immutable
  permission, and `PENDING` status.
- `upload_method: "PUT"`.
- `upload_url`.
- `upload_headers` containing exactly the declared `Content-Type`.
- `upload_expires_at`.

The default presigned PUT lifetime is 10 minutes.

If URL signing fails, the request returns a storage error and the row remains
`PENDING`; the cleanup job later removes the row. No upload URL was exposed to
the client.

### Complete An Upload

`POST /api/files/{file_uuid}/complete/`

Only the owner may call this route. Completion is idempotent for a `READY`
file. For a `PENDING` file, the backend:

1. Calls OSS `HEADObject`.
2. Rejects the completion when the object does not exist.
3. Compares object size with `byte_size`.
4. Compares object `Content-Type` with the declared MIME type.
5. Reads the OSS ETag into the DB record.
6. Atomically transitions the row from `PENDING` to `READY`.
7. Sets `completed_at`.

If the object is missing, or its size or type differs from the declaration, the
row is transitioned to `DELETING` and OSS deletion is attempted before returning
the request failure. If OSS deletion fails, the cleanup job retries.

If a cleanup job already transitioned the row to `DELETING`, completion returns
`410 Gone`. If another request has just transitioned it to `READY`, the endpoint
returns the existing ready file response.

The endpoint returns `200` rather than `201` because the file resource was
already created by the upload request.

### List Files

`GET /api/files/`

Returns paginated files owned by the authenticated caller. The default result
is `READY` files only.

Optional exact filters:

- `status`: `PENDING` or `READY`; `DELETING` is not user-listable.
- `permission`: `PRIVATE` or `PUBLIC_READ`.
- `content_type`
- `created_at_from`
- `created_at_to`

Filters are declared as scalar `Query()` parameters in the router, parsed into
the module `FilterSet`, and applied to the SQL statement. Pagination remains on
the handler.

### Get A File

`GET /api/files/{file_uuid}/`

Returns metadata for an owned `PENDING`, `READY`, or `DELETING` row. It does not
return a new upload URL.

### Get A Private Download URL

`GET /api/files/{file_uuid}/url/`

Returns a short-lived presigned GET URL for an owned `READY` file. The default
lifetime is 5 minutes. The response contains the URL and its expiry timestamp.

The URL is generated with response content type and a UTF-8 content-disposition
based on the stored filename.

### Access A Public File

`GET /api/files/public/{file_uuid}/`

This route does not require authentication. It returns `404` unless both:

- `permission` is `PUBLIC_READ`.
- `status` is `READY`.

It then returns `307 Temporary Redirect` to a short-lived presigned GET URL and
sets `Cache-Control: no-store`.

Public access never lists files. A UUID is required to retrieve public content.

### Delete A File

`DELETE /api/files/{file_uuid}/`

Only the owner may delete a file. The handler atomically transitions a `PENDING`
or `READY` row to `DELETING`, commits, attempts an idempotent OSS delete, and
deletes the DB row when OSS reports success or key-not-found. The endpoint
returns `202 Accepted` after the row is successfully claimed for deletion.

If OSS deletion fails, the row remains `DELETING` and is retried by the cleanup
job. Repeated deletion calls for a row already in `DELETING` return `202`.

## Data Model

The model lives in `src/fivccliche/modules/storages/models.py` and is named
`StorageFile`. Its table name is `storage_file`.

Fields:

- `uuid`: UUID string primary key.
- `user_uuid`: non-null foreign key to `user.uuid`.
- `object_key`: unique, non-null OSS object key.
- `filename`: non-null original display filename.
- `content_type`: non-null approved MIME type.
- `byte_size`: non-null declared size.
- `permission`: `PRIVATE` or `PUBLIC_READ`; immutable after create.
- `status`: `PENDING`, `READY`, or `DELETING`.
- `etag`: nullable OSS ETag, set on successful completion.
- `created_at`: timezone-aware timestamp.
- `updated_at`: timezone-aware timestamp.
- `completed_at`: nullable timezone-aware timestamp.

Indexes:

- `(user_uuid, created_at)`
- `(status, created_at)`
- unique `object_key`

`StorageFile.to_schema()` maps the row to its Pydantic response schema.

All ownership predicates live in the module `FilterSet`, not ad-hoc query
helpers. There are no global storage rows. Regular users can only access their
own rows. This first version does not add a superuser bypass for another user's
private file.

## Storage Service Boundary

The interface lives in `src/fivccliche/services/interfaces/storages.py`:

```python
class IStorageProvider(IComponent):
    @property
    def configured(self) -> bool: ...

    async def presign_put_async(
        self, object_key: str, content_type: str, expires_in: int
    ) -> StorageSignedPut: ...

    async def presign_get_async(
        self,
        object_key: str,
        expires_in: int,
        *,
        content_type: str,
        filename: str,
    ) -> StorageSignedGet: ...

    async def head_async(self, object_key: str) -> StorageHeadObject | None: ...

    async def delete_async(self, object_key: str) -> None: ...
```

`StorageHeadObject` contains byte size, content type, and ETag. `delete_async`
treats an OSS missing-key response as success. OSS transport failures raise a
storage provider exception that routers map to `502 Bad Gateway`.

The Aliyun implementation lives in
`src/fivccliche/services/implements/storages_oss.py`. It uses the official
`oss2` client and runs blocking SDK operations with `asyncio.to_thread`.

The provider is registered against `IStorageProvider` in
`src/fivccliche/settings/services.yml`. The `storages` `IModule` implementation
is registered and included in the module site module list.

The FastAPI dependency resolves the provider. A missing or unconfigured provider
returns `503 Service Unavailable` before a pending DB row is created.

## Configuration

The OSS provider reads an uppercase `OSS` config session:

```json
{
  "OSS": {
    "ACCESS_KEY_ID": "...",
    "ACCESS_KEY_SECRET": "...",
    "ENDPOINT": "oss-cn-hangzhou.aliyuncs.com",
    "BUCKET": "example-bucket",
    "UPLOAD_URL_EXPIRES_SECONDS": 600,
    "DOWNLOAD_URL_EXPIRES_SECONDS": 300
  }
}
```

All fields except the two expiry values are required for the provider to be
configured. Expiry values must be positive. The implementation uses the defaults
above when values are absent or invalid.

The cleanup job reads:

```json
{
  "STORAGE_CLEANUP": {
    "INTERVAL_MINUTES": 60,
    "PENDING_TTL_HOURS": 24,
    "BATCH_SIZE": 100
  }
}
```

`PENDING_TTL_HOURS` must be substantially longer than
`UPLOAD_URL_EXPIRES_SECONDS`; the default is 24 hours. Invalid or non-positive
cleanup values fall back to the defaults.

The bucket and endpoint are never exposed in API responses. Credentials are
never logged.

## Cleanup State Machine

```text
PENDING --complete verified--> READY
PENDING --cleanup expired--> DELETING --object deleted--> row deleted
PENDING --complete rejected--> DELETING --object deleted--> row deleted
READY --owner delete--> DELETING --object deleted--> row deleted
DELETING --retry--> DELETING
```

The `storages-cleanup` module job runs on the existing scheduler.

Each run:

1. Selects at most `BATCH_SIZE` `PENDING` rows older than `PENDING_TTL_HOURS`.
2. Atomically claims each row with an `UPDATE ... WHERE status = 'PENDING'`.
3. Selects a batch of existing `DELETING` rows.
4. Calls idempotent OSS delete for each claimed or already deleting row.
5. Deletes the DB row after OSS reports success or key-not-found.

Cleanup and complete can safely race because both transitions are conditional
on the current `PENDING` status. A late complete after cleanup wins returns
`410 Gone`.

The job reports `config: None` when OSS is not configured, so the scheduler does
not repeatedly run a cleanup worker that cannot delete remote objects.

## Client Upload Flow

1. `POST /api/files/uploads/`.
2. Receive metadata, PUT URL, required headers, and URL expiry.
3. Upload the exact declared bytes directly to OSS with `PUT`.
4. `POST /api/files/{uuid}/complete/`.
5. For private files, call the owner URL route when download access is needed.
6. For public files, share `/api/files/public/{uuid}/`.

The deployment's OSS CORS configuration must allow the browser origin to send
`PUT` requests with a `Content-Type` request header. OSS must expose the `ETag`
response header if the frontend uses it.

The presigned PUT URL does not by itself enforce the 20 MiB limit. A malicious
client may upload a larger object. Completion rejects it and moves the file to
`DELETING`; the cleanup workflow removes the rejected object if the immediate
delete attempt fails.

## Error Handling

| Condition | Response |
|---|---|
| Invalid filename/type/size/permission | `422` from request validation |
| Storage provider missing or not configured | `503` |
| Another user's file | `404` |
| Complete on another user's file | `404` |
| Complete before upload | `400` |
| Complete with size/type mismatch | `400` |
| Complete after cleanup claimed row | `410` |
| OSS HEAD/delete failure | `502`, except missing key on delete is success |
| URL request for pending/deleting file | `409` |
| Public access to private or missing file | `404` |

Unexpected storage exceptions are logged with the file UUID and object key, but
without credentials or signed URLs.

## Testing

API tests use a fake `IStorageProvider` and do not access Aliyun:

- Create validates filename, MIME/suffix pairs, 20 MiB maximum, and permission.
- Create commits `PENDING` before returning an upload URL.
- Complete succeeds only when HEAD size and type match.
- Complete is idempotent for `READY`.
- Complete rejects missing or mismatched objects and transitions to `DELETING`.
- Complete races with cleanup using the conditional status transition.
- Lists filter by status, permission, type, and created date.
- Private URL access enforces ownership and `READY` state.
- Public route requires `PUBLIC_READ` and `READY`.
- Permission cannot be changed through the HTTP API.
- Delete claims `PENDING`/`READY` rows and retries existing `DELETING` rows.

Job tests cover configuration defaults, stale pending selection, conditional
claims, batch limits, missing-key idempotency, and DB deletion after object
deletion.

Provider tests inject a small signer/client boundary to verify configuration
handling, URL expiry, signed headers, HEAD result mapping, and exception
mapping. Real OSS integration is manual deployment verification and is not part
of automated tests.

Run:

```bash
make test
make lint
make typecheck
```

## Documentation Updates During Implementation

- Extend `docs/architecture.md` with the `storages` module and service.
- Add a focused `docs/storages.md` covering configuration, API flow, cleanup,
  and OSS CORS prerequisites.
- Update `CLAUDE.md` module and service lists.
