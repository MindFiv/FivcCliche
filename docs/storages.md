# Storages

The `storages` module stores user file metadata in PostgreSQL and uploads file
bytes directly to Aliyun OSS with presigned URLs. Run `fivccliche migrate`
after upgrading so the `storage_file` table is created.

If you already have a `storage_file` table from an earlier revision, recreate it
or alter `permission` → `is_public` (boolean) yourself; this project uses
`create_all` and does not ship Alembic migrations.

## Configuration

```json
{
  "OSS": {
    "ACCESS_KEY_ID": "...",
    "ACCESS_KEY_SECRET": "...",
    "ENDPOINT": "oss-cn-hangzhou.aliyuncs.com",
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

`IStorageProvider.get_storage(bucket_name)` returns an `IStorage` for that
bucket (shared credentials/endpoint). Incomplete OSS credentials yield `None`
(HTTP `503`). Upload/download URL lifetimes come from the config session;
`get_storage(..., upload_expires_in=, download_expires_in=)` can override them.

`is_public` is immutable after upload. Public uploads include
`x-oss-object-acl: public-read` on the presigned PUT; private uploads use
`private`. Authenticated `GET /api/files/{uuid}/url/` returns a signed URL for
private files and an unsigned object URL (`expires_at: null`) for public ones.
There is no unauthenticated public route.

## Upload flow

1. `POST /api/files/uploads/` validates filename, MIME type, size, and
   `is_public`, creates a `PENDING` row, and returns a presigned PUT URL plus
   required headers (`Content-Type` and `x-oss-object-acl`).
2. The client uploads directly to OSS with the returned method, URL, and
   headers.
3. `POST /api/files/{uuid}/complete/` calls OSS HEAD, verifies size and type,
   stores the ETag, and transitions the row to `READY`.
4. Owners use `GET /api/files/{uuid}/url/` (always authenticated).

Allowed files are images, PDF, text, audio, and video up to 20 MiB. Exact
suffix/MIME pairs live in `StorageFileCreateSchema`.

Expired pending rows and owner-deleted rows move to `DELETING`. The
`storages-cleanup` job deletes the OSS object and then the DB row; missing OSS
keys count as successful deletes. Failed deletes remain `DELETING` for retry.

OSS CORS must allow client-origin `PUT` requests with `Content-Type` and
`x-oss-object-acl`. Automated tests use a fake storage and do not contact OSS.
