# Security

## Reporting a vulnerability

Report vulnerabilities through GitHub's private reporting form: open the repository's **Security** tab and choose **Report a vulnerability**. Please do not open a public issue for a security problem.

A useful report says which version or commit is affected, how to reproduce the problem, and what an attacker gains.

## Scope

The parts of this project that enforce a security property are:

- the access filter inside retrieval (`src/governed_rag/retrieve.py`), which decides which documents a reader can see;
- the HTTP service (`src/governed_rag/service.py`), which maps a bearer token to groups and never reads groups from the request;
- the prompt and the verification gates, which treat retrieved text as data.

## Deployment notes

- `compose.yaml` starts a development database with the user and password `rag`. Do not expose it, and do not reuse those credentials.
- `rag serve` speaks plain HTTP and binds to 127.0.0.1 by default. Put TLS and rate limiting in front of it.
- The token file given to `rag serve --tokens` holds credentials. Keep it out of version control; `.gitignore` already excludes `tokens.json`.
