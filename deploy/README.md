# Azure PostgreSQL Deployment

## Connection String Format

```
postgresql+psycopg://USER:PASS@SERVER.postgres.database.azure.com:5432/robot_agent?sslmode=require
```

- `sslmode=require` is non-optional; Azure rejects unencrypted connections.
- Replace `USER`, `PASS`, and `SERVER` with your Azure PostgreSQL Flexible Server credentials.

## Local Postgres (Docker) for Parity Testing

```bash
docker compose -f deploy/docker-compose.dev.yml up -d
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent
uv run alembic upgrade head
uv run python -m app.robot_agent.seed.run
```

## Firewall Rules

- Allow the deploying host's outbound IP address in the Azure PostgreSQL firewall.
- For production, use VNet integration instead of public IP whitelisting.

## Secret Handling

- `DATABASE_URL` must never be stored in the repository.
- For local development, use a `.env` file (`.gitignore` already covers it).
- For Azure App Service, set `DATABASE_URL` in **App Service Configuration → Application settings** or in **Azure Key Vault** with a managed identity reference.

## Migration on Deploy

- Run `alembic upgrade head` as a **pre-start hook** (not on every container start) to avoid migration race conditions.
- For Azure App Service, add the command to a startup script or use the `PRE_BUILD_COMMAND` / `POST_BUILD_COMMAND` hooks.

## Server SKU Recommendation

- Burstable B1ms is too small for sustained simulation load.
- For production, use **GP_Standard_D2s_v3** or higher.
