# Authentication

Nimbus API uses API keys for authentication. Every request must include an `Authorization: Bearer <API_KEY>` header. API keys are generated from the dashboard under Settings > API Keys, and each key can be scoped to read-only or read-write permissions.

Keys are prefixed with `nb_live_` for production and `nb_test_` for the sandbox environment. Sandbox keys never touch production data and are safe to share with contractors.

## Rotating keys

We recommend rotating API keys every 90 days. When you generate a new key, the old key remains valid for 24 hours to allow a graceful cutover — after that window it is automatically revoked. There is no way to recover a revoked key; you must generate a new one.

## OAuth2 for user-facing apps

If you're building an application on behalf of end users (rather than server-to-server), use the OAuth2 authorization code flow instead of a static API key. Register your app under Settings > OAuth Apps to receive a client ID and client secret. Access tokens issued via OAuth2 expire after 1 hour; use the refresh token to obtain a new one without re-prompting the user.

## Common authentication errors

- `401 invalid_api_key`: the key is malformed, revoked, or belongs to a different environment (e.g. using a test key against the production endpoint).
- `403 insufficient_scope`: the key is valid but lacks permission for the requested operation, most often a read-only key attempting a write.
