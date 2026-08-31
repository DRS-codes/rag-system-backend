# Data Retention & Deletion

Nimbus retains uploaded documents and derived embeddings for as long as your account is active. Request logs (metadata only, not payloads) are retained for 30 days for debugging and abuse prevention, then permanently deleted.

## Deleting your data

Deleting a document via `DELETE /v1/documents/{id}` removes it and all derived chunks/embeddings immediately from the primary database. Backups are retained for disaster recovery for up to 14 days after deletion, after which the data is unrecoverable even by Nimbus staff.

## Account deletion

Deleting your account triggers deletion of all associated documents, API keys, and OAuth tokens within 24 hours. Billing records are retained for 7 years to comply with financial recordkeeping regulations, even after account deletion — this is the one exception to full data removal.

## Data residency

By default, data is stored in the US region. Enterprise customers can request EU data residency, which pins storage and processing to EU infrastructure for GDPR compliance. Data residency cannot be changed retroactively for an existing account without a full re-ingestion of documents.

## Third-party subprocessors

Nimbus uses subprocessors for embedding generation and object storage; a current list is published at nimbus.example/subprocessors and customers are notified 30 days before any new subprocessor is added.
