# FinScope AI — Alembic Migrations

A generic, single database configuration.

## Setup
```bash
alembic init alembic  # Already done
alembic revision --autogenerate -m "Initial tables"
alembic upgrade head
```
