-- Creates the role FinVault's application connects as.
--
-- Mounted into the Postgres container's /docker-entrypoint-initdb.d, so it
-- runs once when the data directory is first initialized. On an existing
-- volume it does not re-run — apply it by hand there.
--
-- **Why a second role at all.** The `finvault` account created by
-- POSTGRES_USER is a superuser, and a superuser ignores Row Level Security
-- completely: policies are installed, pg_policies lists them, and not one
-- row is ever filtered. Connecting the application as that account means
-- tenant isolation is switched on and enforcing nothing, with no visible
-- difference from the inside. This role exists so that cannot happen.
--
-- Two properties, both load-bearing:
--   1. NOSUPERUSER, NOBYPASSRLS — policies apply to it.
--   2. It does not OWN the tables — plain ENABLE would exempt an owner, and
--      relying on FORCE alone means one missed ALTER TABLE silently disables
--      isolation.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'finvault_app') THEN
    CREATE ROLE finvault_app
      LOGIN
      PASSWORD 'finvault_app_dev_only'
      NOSUPERUSER
      NOCREATEDB
      NOCREATEROLE
      NOBYPASSRLS;
  END IF;
END
$$;

GRANT CONNECT ON DATABASE finvault TO finvault_app;
GRANT USAGE ON SCHEMA public TO finvault_app;

-- DML only. The application reads and writes rows; it does not create,
-- alter, or drop tables — schema changes are a migration's job, run by the
-- owner. Withholding DDL here also means the app can never grant itself an
-- exemption from the policies it is subject to.
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO finvault_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO finvault_app;

-- Same grants for tables created later, so a new table added by a migration
-- is usable without remembering to re-run this file. Note the default
-- privileges belong to the role that creates the objects.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO finvault_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO finvault_app;
