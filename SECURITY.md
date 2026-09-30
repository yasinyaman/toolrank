# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, through GitHub's private vulnerability reporting: the
**Report a vulnerability** button on this repository's **Security** tab. Do not open a public issue.

We aim to acknowledge a report within three working days and to agree on a disclosure date with
you.

## Supported versions

toolrank is not yet at 1.0: fixes go into the latest release only.

## Defaults worth knowing

- `toolrank serve` listens on 127.0.0.1 unless told otherwise, and refuses any other address
  without an API key (`--api-key`, `TOOLRANK_API_KEY` or `--api-keys`). It speaks plain HTTP: beyond
  the machine itself, run it behind a proxy that terminates TLS, or the key travels in the clear.
- `/v1` checks the Host and Origin headers, so a web page cannot reach a server on your machine;
  `/mcp` has the MCP SDK's DNS-rebinding protection with the same allowed hosts.
- OpenAPI tools send only GET and HEAD requests unless you pass `--allow-write`. Headers from the
  config (credentials) go only to that source's `base_url`, and redirects are not followed.
- The embedding endpoint gets `TOOLRANK_EMB_API_KEY` if you set one; `OPENAI_API_KEY` goes only to
  `https://api.openai.com`, and no key is carried onto a redirect.
- The usage log keeps requests and arguments as keyed digests, not text, unless `--log-text`; tool
  names, scores and the server's own instruction are text.
- Packaged heads load with `allow_pickle=False`, and downloads are checked against a sha256.
- A tool's output goes to the agent's model. Treat tools like any other input that can steer a
  model, and keep write access for tools you trust.
