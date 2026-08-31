# Rate Limits

Nimbus API enforces rate limits per API key, not per account, so an application with multiple keys gets independent limits per key.

## Default limits

- Free tier: 60 requests per minute, 5,000 requests per day.
- Growth tier: 600 requests per minute, 200,000 requests per day.
- Enterprise tier: custom limits negotiated per contract, typically starting at 5,000 requests per minute.

When a limit is exceeded, the API responds with `429 rate_limit_exceeded` and a `Retry-After` header indicating how many seconds to wait before retrying. We strongly recommend implementing exponential backoff rather than retrying immediately on a 429, since immediate retries during a rate-limited burst tend to make the backlog worse.

## Burst allowance

All tiers get a short burst allowance of 2x the per-minute limit for up to 10 seconds, to absorb traffic spikes without hard-failing legitimate short bursts. Burst usage counts against the daily quota as normal.

## Checking your current usage

Every response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers so you can monitor usage proactively rather than waiting for a 429.

## Requesting a limit increase

Growth and Enterprise customers can request a temporary or permanent rate limit increase by contacting support with expected peak traffic. Increases are typically provisioned within 1 business day.
