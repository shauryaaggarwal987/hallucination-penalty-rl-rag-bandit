# Security

Never commit API keys or credentials. Supply the Gemini key through the `GOOGLE_API_KEY` environment variable or a local `.env` file. `.env` and common caches are ignored by Git.

The public-release copy was scanned for credential literals, bearer tokens, private keys, password assignments, and secret assignments. Environment-variable names and placeholder values are expected; real credential values are not.

If a credential is accidentally committed, revoke it with its provider before removing it from Git history.

