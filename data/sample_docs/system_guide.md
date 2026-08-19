# Cloud Infrastructure & Microservices Architecture

## Authentication Service
Our authentication service uses JWT (JSON Web Tokens) with RS256 asymmetric encryption. 
Access tokens expire in 15 minutes, while refresh tokens are valid for 7 days.
To rotate authentication secrets, update the `AUTH_JWT_SECRET` environment variable and restart the auth container.

## Database Configurations
We use PostgreSQL 15 for relational transactional storage.
To execute migrations safely:
1. Run `alembic upgrade head`
2. Check schema integrity via PgAdmin or `psql` CLI.
All connection pools are capped at a maximum of 20 concurrent connections per replica.

## Error Codes Reference
- `ERR_AUTH_EXPIRED_401`: Token timestamp has exceeded the 15-minute validity window.
- `ERR_DB_TIMEOUT_504`: Database connection pool exhausted or query execution exceeded 30 seconds.
- `ERR_RATE_LIMIT_429`: IP request frequency exceeded 100 requests per minute.