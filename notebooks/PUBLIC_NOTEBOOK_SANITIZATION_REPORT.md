# Public Notebook Sanitization Report

The authoritative notebook release copy was checked for credential literals, bearer tokens, private-key blocks, password/secret assignments, `.env` content, and sensitive administrative information. No secret value was found or redacted.

It was also checked for third-party corpus-text redistribution. One saved output containing a long text-bearing corpus dump was removed from the public copy. Code, Markdown, plots, metrics, result tables, and non-text scientific outputs were preserved. The original notebook was not modified.

