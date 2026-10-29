# Storages

The `storages` module stores user file metadata in PostgreSQL and uploads file
bytes with client-direct presigned URLs. The default object backend is **S3**
(`aioboto3`), including S3-compatible endpoints (MinIO, Ceph RGW, and similar)
via `ENDPOINT`. Aliyun OSS remains available: change the `IStorageProvider`
class in `src/fivccliche/settings/services.yml` to
`fivccliche.services.implements.storages_oss.OssStorageProvider` and use the
`OSS` config session instead.

Run `fivccliche migrate` after upgrading so `storage_folder` / `storage_file`
tables exist. If you already have tables from an earlier revision, recreate them
or alter columns yourself (`folder_uuid`, relative `object_key`, `filename` →
`name`, and the `storage_folder` table). This project uses `create_all` and does
not ship Alembic migrations. Older rows with `object_key=storages/{uuid}` are
incompatible with the current relative-key + prefix model.

## Configuration (S3)

```json
{
  "S3": {
    "ACCESS_KEY_ID": "...",
    "ACCESS_KEY_SECRET": "...",
    "ENDPOINT": "https://minio.example.com",
    "REGION": "us-east-1",
    "BUCKET": "example-bucket",
    "UPLOAD_URL_EXPIRES_SECONDS": 600,
    "DOWNLOAD_URL_EXPIRES_SECONDS": 300
  },
  "STORAGE_CLEANUP": {
    "INTERVAL_MINUTES": 60,
    "PENDING_TTL_HOURS": 24,
    "BATCH_SIZE": 100
  }
}
```

- `ENDPOINT` may be empty for AWS S3 defaults; set it for third-party compatible
  APIs (`endpoint_url` in aioboto3).
- `REGION` defaults to `us-east-1`.
- Incomplete credentials yield `get_storage` → `None` (HTTP `503`).
- `get_storage()` (no name) uses the configured default `BUCKET`; pass an
  explicit bucket name to select another.
- `get_storage(..., key_prefix=)` scopes all object operations under a bucket
  subdirectory (leading/trailing `/` are stripped).
- `get_storage(..., upload_expires_in=, download_expires_in=)` take
  `timedelta` overrides for URL lifetimes; omit them to use config seconds.

### Aliyun OSS (optional)

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

Only one provider class is registered at a time.

## Folders

`StorageFolder` binds a `bucket_name` + `bucket_prefix` so one physical bucket
can be split into isolated `IStorage` views.

- HTTP: `/api/folders/`
- Create / update / delete: **superuser only** (create always stores
  `user_uuid=None`)
- List / get: Readable (owned or global); ordinary users can see global folders
- Upload may pass optional `folder_uuid` (must be Readable)

## Ownership and object keys

Same user-or-global rules as agent configs:

- Superuser create → `user_uuid=None` (global / shared)
- Regular create → caller's `user_uuid`
- List/get/url: owned rows plus globals (Readable)
- Complete/delete: owned only for regular users; superusers may also write globals
  (Editable)
- List filter `is_global=true` → `user_uuid IS NULL`; `false` → not null; omit to
  skip

DB `object_key` is always the **relative** key (`{file_uuid}`). `name` is a
client-supplied display name used for MIME validation; it is **not** unique and
does not appear in the object key.

`IStorage` resolution:

- With folder → `get_storage(folder.bucket_name, key_prefix=folder.bucket_prefix)`
- Without folder → `get_storage(None, key_prefix=users/{user_uuid})` for owned
  files, or empty prefix for globals (`user_uuid is None`)

Backend Key examples: `users/{user_uuid}/{file_uuid}` or
`{folder.bucket_prefix}/{file_uuid}`.

`is_public` is immutable after upload and is independent of ownership. Public
uploads include `x-amz-acl: public-read` (S3) or `x-oss-object-acl: public-read`
(OSS) on the presigned PUT; private uploads use `private`. On AWS buckets with
**Object Ownership: Bucket owner enforced**, object ACLs are ignored—use a
bucket policy if public objects are required. Authenticated
`GET /api/files/{uuid}/url/` returns a signed URL for private files and an
unsigned object URL (`expires_at: null`) for public ones. There is no
unauthenticated public route.

## Upload flow

1. `POST /api/files/uploads/` validates `name`, MIME type, size,
   `is_public`, and optional `folder_uuid`, creates a `PENDING` row, and returns
   a presigned PUT URL plus required headers (`Content-Type` and ACL header).
2. The client uploads directly to the object store with the returned method,
   URL, and headers.
3. `POST /api/files/{uuid}/complete/` HEADs the object, verifies size and type,
   stores the ETag, and transitions the row to `READY`.
4. Readable callers use `GET /api/files/{uuid}/url/` (always authenticated).

Allowed files are images, PDF, text, audio, and video up to 20 MiB. Exact
suffix/MIME pairs live in `StorageFileCreateSchema`.

Expired pending rows and owner-deleted rows move to `DELETING`. The
`storages-cleanup` job resolves `IStorage` per row (folder or default prefix),
deletes the object, then the DB row; missing keys count as successful deletes.
Failed deletes remain `DELETING` for retry.

CORS must allow client-origin `PUT` with `Content-Type` and the ACL header.
Automated tests use a fake storage and do not contact a real bucket.
