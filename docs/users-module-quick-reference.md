# Users Module - Quick Reference

## File Structure
```
src/fivccliche/
├── database.py                    # Database configuration
└── modules/users/
    ├── __init__.py               # Module registration
    ├── models.py                 # SQLModel User model
    ├── schemas.py                # Pydantic schemas
    ├── services.py               # Business logic
    └── routers.py                # API endpoints

tests/
├── test_users_service.py         # Unit tests (14 tests)
└── test_users_api.py             # Integration tests (12 tests)
```

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/users/` | Create new user | None |
| GET | `/users/` | List users (paginated) | None |
| GET | `/users/self` | Get authenticated user's profile | Bearer Token |
| GET | `/users/{user_id}` | Get user by ID | Admin |
| DELETE | `/users/{user_id}` | Delete user | Admin |
| POST | `/users/login` | Authenticate user and return JWT token | None |

## Key Classes

### UserService
```python
# Password operations
UserService.hash_password(password: str) -> str
UserService.verify_password(plain: str, hashed: str) -> bool

# CRUD operations
UserService.create_user(session, user_create)
UserService.get_user_by_uuid(session, uuid)
UserService.get_user_by_username(session, username)
UserService.get_user_by_email(session, email)
UserService.get_all_users(session, skip=0, limit=100)
UserService.update_user(session, user, user_update)
UserService.delete_user(session, user)

# Authentication
UserService.authenticate_user(session, username, password)
```

## Database Models

### User
- `id` (str, PK): Unique identifier (UUID string)
- `username` (str, unique): User login name
- `email` (str, unique, nullable): User email (optional)
- `hashed_password` (str): Argon2 hash
- `created_at` (datetime): Creation timestamp
- `signed_in_at` (datetime, nullable): Last login
- `is_active` (bool): Account status
- `is_superuser` (bool): Admin flag

## Request/Response Examples

### Create User
**Request:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2024-01-15T10:30:00",
  "signed_in_at": null,
  "is_active": true,
  "is_superuser": false
}
```

### Create User Without Email
**Request:**
```json
{
  "username": "jane_doe",
  "password": "SecurePass123!"
}
```

**Response (201):**
```json
{
  "id": "660f9500-f39c-52e5-b827-557766551111",
  "username": "jane_doe",
  "email": null,
  "created_at": "2024-01-15T10:35:00",
  "signed_in_at": null,
  "is_active": true,
  "is_superuser": false
}
```

### Login
**Request:**
```json
{
  "username": "john_doe",
  "password": "SecurePass123!"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

## Error Responses

| Status | Error | Cause |
|--------|-------|-------|
| 400 | Username already registered | Duplicate username |
| 400 | Email already registered | Duplicate email |
| 401 | Invalid username or password | Auth failed |
| 404 | User not found | UUID doesn't exist |

## Running Tests

```bash
# All tests
uv run pytest tests/test_users_*.py -v

# Unit tests only
uv run pytest tests/test_users_service.py -v

# Integration tests only
uv run pytest tests/test_users_api.py -v

# With coverage
uv run pytest tests/test_users_*.py --cov=src/fivccliche/modules/users
```

## Configuration

Database URL via environment variable:
```bash
export DATABASE_URL="sqlite:///./fivccliche.db"
# or for PostgreSQL:
export DATABASE_URL="postgresql://user:pass@localhost/dbname"
```

## Security Notes
- Passwords hashed with Argon2 (resistant to GPU attacks)
- No password in API responses
- Email validation enforced when provided
- Email is optional (users can be created without email)
- Unique constraints on username/email
- Proper HTTP status codes for errors

